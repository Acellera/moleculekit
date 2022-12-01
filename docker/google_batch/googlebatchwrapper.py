from google.cloud import batch_v1
from google.cloud.batch_v1.types.job import JobStatus
from google.cloud import storage
from google.oauth2 import service_account
import os
from glob import glob
import uuid
from moleculekit.util import ensurelist
from protocolinterface import ProtocolInterface, val
from tqdm import tqdm
import logging
import time

logger = logging.getLogger("batch_test")

# pip install google-cloud-storage google-cloud-batch


def create_container_job(
    project_id: str,
    region: str,
    job_name: str,
    container_name: str,
    bucket_name: str,
    inputdir: str,
    task_count: int,
    parallelism: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    max_run_duration: str,
    provisioning_model: str,
    cpu_milli: int,
    memory_mib: int,
) -> batch_v1.Job:
    # Define what will be done as part of the job.
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = (
        f"{region}-docker.pkg.dev/{project_id}/containers/{container_name}:v1"
    )
    runnable.container.entrypoint = "/bin/bash"
    cmds = ["/scripts/execute.sh"]
    if inputdir is not None and len(inputdir):
        cmds.append(inputdir)
    runnable.container.commands = cmds
    runnable.container.volumes = [f"/mnt/disks/{bucket_name}:/workdir"]
    if accelerator_type is not None:
        runnable.container.volumes += [
            "/var/lib/nvidia/lib64:/usr/local/nvidia/lib64",
            "/var/lib/nvidia/bin:/usr/local/nvidia/bin",
        ]
        runnable.container.options = "--privileged"

    # Jobs can be divided into tasks. In this case, we have only one task.
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    # Mount bucket for task
    gcs = batch_v1.GCS()
    gcs.remote_path = bucket_name
    vol = batch_v1.Volume()
    vol.gcs = gcs
    vol.mount_path = f"/mnt/disks/{bucket_name}"

    # We can specify what resources are requested by each task.
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = cpu_milli  # in milliseconds per cpu-second. This means the task requires 2 whole CPUs.
    resources.memory_mib = memory_mib  # in MiB
    task.compute_resource = resources

    task.max_retry_count = 2
    task.max_run_duration = max_run_duration
    task.volumes = [vol]

    # Tasks are grouped inside a job using TaskGroups.
    # Currently, it's possible to have only one task group.
    group = batch_v1.TaskGroup()
    group.task_count = task_count
    group.parallelism = parallelism
    group.task_spec = task

    # Policies are used to define on what kind of virtual machines the tasks will run on.
    # In this case, we tell the system to use "e2-standard-4" machine type.
    # Read more about machine types here: https://cloud.google.com/compute/docs/machine-types
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = machine_type
    if accelerator_type is not None:
        accelerator = batch_v1.AllocationPolicy.Accelerator()
        accelerator.type_ = accelerator_type
        accelerator.count = accelerator_count
        policy.accelerators = [accelerator]
    if provisioning_model:
        mapping = {
            "unspecified": batch_v1.AllocationPolicy.ProvisioningModel.PROVISIONING_MODEL_UNSPECIFIED,
            "standard": batch_v1.AllocationPolicy.ProvisioningModel.STANDARD,
            "spot": batch_v1.AllocationPolicy.ProvisioningModel.SPOT,
            "preemptible": batch_v1.AllocationPolicy.ProvisioningModel.PREEMPTIBLE,
        }
        policy.provisioning_model = mapping[provisioning_model]

    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    if accelerator_type is not None:
        instances.install_gpu_drivers = True
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.labels = {"env": "testing", "type": "container"}
    # We use Cloud Logging as it's an out of the box available option
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name + "-" + str(uuid.uuid4())[:8]
    # The job's parent is the region in which the job will run
    create_request.parent = f"projects/{project_id}/locations/{region}"

    return create_request


def create_bucket(storage_client, bucket_name):
    buckets = storage_client.list_buckets()
    bucket_names = [b.name for b in buckets]
    if bucket_name not in bucket_names:
        bucket = storage_client.create_bucket(bucket_name)
    else:
        bucket = storage_client.bucket(bucket_name)

    return bucket


def upload_to_bucket(bucket, source, dest):
    if os.path.isdir(source):
        files = []
        for ff in glob(os.path.join(source, "**", "*"), recursive=True):
            if not os.path.isdir(ff):
                files.append(ff)

        for ff in tqdm(files, desc=f"Uploading files to {bucket}/{dest}"):
            relname = os.path.relpath(ff, source)
            target = f"{dest}{relname}"
            blob = bucket.blob(target)
            blob.upload_from_filename(ff)
    else:
        basename = os.path.basename(source)
        target = f"{dest}{basename}"
        print(f"Uploading {source} to {target}")
        blob = bucket.blob(target)
        blob.upload_from_filename(source)


def _download(bucket, source, target):
    blob = bucket.blob(source)
    if not blob.exists():
        raise RuntimeError(f"{source} does not exist.")
        return
    blob.download_to_filename(target)


def _get_files_recursive(bucket, source, files):
    blobs = bucket.list_blobs(prefix=source, delimiter="/")
    # Order is important here. The prefixes variable is not populated until we iterate over blobs
    for blob in blobs:
        if not blob.name.endswith("/"):
            files.append(blob.name)

    for prefix in blobs.prefixes:
        _get_files_recursive(bucket, prefix, files)

    files = sorted(files)
    return files


def download_from_bucket(bucket, source, target):
    directory_mode = False

    if len(source) == 0 or source.endswith("/"):
        directory_mode = True
        files = _get_files_recursive(bucket, source, [])
    else:
        files = [source]

    for ff in tqdm(files, desc="Downloading files"):
        dest = target
        relname = os.path.relpath(ff, source)
        if directory_mode or os.path.isdir(target):
            dest = os.path.join(target, relname)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _download(bucket, ff, dest)


class GoogleBatchSession:
    def __init__(self, credential_file, project, region) -> None:
        self.project = project
        self.region = region
        cred = service_account.Credentials.from_service_account_file(credential_file)
        self.storage_client = storage.Client(project=project, credentials=cred)
        self.batch_client = batch_v1.BatchServiceClient.from_service_account_file(
            credential_file
        )

    def create_container_job2(
        self,
        container,
        bucket_name,
        job_name_prefix,
        inputdir,
        task_count,
        parallelism,
        machine_type,
        accelerator_type,
        accelerator_count,
        max_run_duration,
        provisioning_model,
        cpu_milli,
        memory_mib,
    ):
        create_request = create_container_job(
            self.project,
            self.region,
            job_name_prefix,
            container,
            bucket_name,
            inputdir,
            task_count,
            parallelism,
            machine_type,
            accelerator_type,
            accelerator_count,
            max_run_duration,
            provisioning_model,
            cpu_milli,
            memory_mib,
        )
        job = self.batch_client.create_job(create_request)
        return job.name

    def get_job_status(self, job_name):
        get_request = batch_v1.GetJobRequest(name=job_name)
        job = self.batch_client.get_job(request=get_request)
        return job.status.state

    def create_bucket(self, bucket_name):
        buckets = self.storage_client.list_buckets()
        bucket_names = [b.name for b in buckets]
        if bucket_name in bucket_names:
            logger.warning(
                f"Bucket {bucket_name} already exists. Please provide different bucket name or delete the bucket with session.delete_bucket('{bucket_name}')"
            )
            return self.storage_client.get_bucket(bucket_name)

        bucket = self.storage_client.create_bucket(bucket_name)
        return bucket

    def delete_bucket(self, bucket_name):
        blobs = self.storage_client.list_blobs(bucket_name)
        for blob in tqdm(blobs, desc="Deleting bucket files"):
            blob.delete()
        # Now that all the files have been deleted, we can delete our empty bucket
        bucket = self.storage_client.get_bucket(bucket_name)
        bucket.delete()


class GoogleBatchJob(ProtocolInterface):
    def __init__(
        self,
        session: GoogleBatchSession,
        bucket_name: str,
        job_name_prefix: str = "job",
        job_name: str = None,
    ) -> None:
        super().__init__()
        self._session = session

        self._job_name = job_name
        self._job_name_prefix = job_name_prefix
        self._arg(
            "container", "string", "Container in which to run", None, val.String()
        )
        self._arg("inputpath", "string", "Input path", None, val.String())
        self._arg("remotepath", "string", "Remote path", "", val.String())
        self._arg(
            "ncpu",
            "int",
            "Number of cpus on each VM",
            1,
            val.Number(int, "POS"),
        )
        self._arg(
            "memory",
            "int",
            "Memory (RAM) on each VM in Gigabytes",
            4,
            val.Number(int, "POS"),
        )
        self._arg(
            "parallelism",
            "int",
            "Number of jobs to run in parallel",
            None,
            val.Number(int, "POS"),
        )
        self._arg(
            "machine_type",
            "string",
            "GCP machine type (i.e. e2-standard-4, a2-highgpu-1g)",
            None,
            val.String(),
        )
        self._arg(
            "accelerator_type",
            "string",
            "Accelerator type (i.e. nvidia-tesla-k80, nvidia-tesla-a100)",
            None,
            val.String(),
        )
        self._arg(
            "accelerator_count",
            "int",
            "Number of accelerators to attach to the compute nodes",
            1,
            val.Number(int, "POS"),
        )
        self._arg(
            "max_run_duration",
            "string",
            "Maximum run duration of a job",
            "3600s",
            val.String(),
        )
        self._arg(
            "provisioning_model",
            "string",
            "The VM provisioning model: ['standard', 'spot', 'unspecified', 'preemptible']. Default: 'spot'",
            "spot",
            val.String(),
        )

        self._bucket = self._session.create_bucket(bucket_name)
        self._bucket_name = bucket_name

    def get_status(self, _logger=True):
        """Prints or returns the job status

        Parameters
        ----------
        _logger: bool
            Set as True for printing the info and errors in sys.stdout.
            If False, it returns the same information (default=True).

        Returns
        -------
        status : JobStatus
            The current status of the job
        """
        if self._job_name is None:
            raise RuntimeError("No job submitted yet")
        status = self._session.get_job_status(self._job_name)
        if _logger:
            logger.info(f"Job Status: {str(status)}")

        return status

    def submit(self):
        if self.container is None:
            raise RuntimeError("Please specify a container")
        if self.inputpath is None:
            raise RuntimeError("Please specify a inputpath")
        if self.remotepath is None:
            raise RuntimeError("Please specify a remotepath")
        if len(self.remotepath) and not self.remotepath.endswith("/"):
            raise RuntimeError("Remotepath must end with /")
        if self.machine_type is None:
            raise RuntimeError("Please specify the machine type to run on")

        folders = glob(os.path.join(self.inputpath, "*", ""))
        for ff in folders:
            if not os.path.exists(os.path.join(ff, "run.sh")):
                raise RuntimeError(f"No run.sh file exists in {ff}")

        upload_to_bucket(self._bucket, self.inputpath, self.remotepath)
        self._job_name = self._session.create_container_job2(
            self.container,
            self._bucket_name,
            self._job_name_prefix,
            self.remotepath,
            len(folders),
            self.parallelism,
            self.machine_type,
            self.accelerator_type,
            self.accelerator_count,
            self.max_run_duration,
            self.provisioning_model,
            self.ncpu * 1000,
            self.memory * 1000,
        )
        logger.info(f"Submitted job with name: {self._job_name}")

    def retrieve(self, path):
        remote = self.remotepath
        download_from_bucket(self._bucket, remote, path)

        status = self.get_status(_logger=False)
        if status in (JobStatus.State.SUCCEEDED, JobStatus.State.FAILED):
            # Cleaning up job
            self._session.delete_bucket(self._bucket_name)
            try:
                request = batch_v1.DeleteJobRequest(name=self._job_name)
                operation = self._session.batch_client.delete_job(request)
                # response = operation.result() # makes it wait until deletion
            except Exception:
                pass

    def wait(
        self,
        on_status=(JobStatus.State.SUCCEEDED, JobStatus.State.FAILED),
        seconds=10,
        _logger=True,
    ):
        """Blocks execution until the job has reached the specific status or any of a list of statuses.

        Parameters
        ----------
        on_status : JobStatus or list of JobStatus
            The status(es) at which the job will not be waited upon and the code execution will continue.
        seconds : float
            The sleep time between status cheching
        _logger: bool
            Set to False to reduce verbosity
        """
        on_status = ensurelist(on_status)

        while True:
            status = self.get_status(_logger=False)
            if status in on_status:
                break

            if _logger:
                logger.info(
                    f"Job status {str(status)}. Waiting for it to reach ({', '.join(map(str, on_status))}). Sleeping for {seconds} seconds."
                )
            time.sleep(seconds)
        logger.info(f"Job status {str(status)}")

    def cancel(self):
        raise NotImplementedError("Job cancelling has not been implemented yet")


if __name__ == "__main__":
    # session = GoogleBatchSession(credential_file, project, "us-central1")
    # job = GoogleBatchJob(session, "testbucket-moleculekit-2")
    # job.container = "moleculekit-service"
    # job.inputpath = "./test2/"
    # job.parallelism = 2
    # job.machine_type = "e2-standard-4"
    # job.provisioning_model = "standard"  # Standard VM provision
    # job.submit()
    # job.wait()
    # job.retrieve("./output/")

    # session = GoogleBatchSession(credential_file, project, "us-central1")
    # job = GoogleBatchJob(session, "testbucket-moleculekit-3")
    # job.container = "acemd-service"
    # job.inputpath = "./test3/"
    # job.parallelism = 2
    # job.machine_type = "n1-standard-2"
    # job.accelerator_type = "nvidia-tesla-k80"  # gcloud compute accelerator-types list
    # job.accelerator_count = 1
    # job.provisioning_model = "spot"  # Spot pricing
    # job.submit()
    # job.wait()
    # job.retrieve("./output/")

    pass
