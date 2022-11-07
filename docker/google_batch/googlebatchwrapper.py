from google.cloud import batch_v1
from google.cloud.batch_v1.types.job import JobStatus
from google.cloud import storage
from google.oauth2 import service_account
import os
from glob import glob
import uuid
from moleculekit.util import ensurelist
from protocolinterface import ProtocolInterface, val
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
    runscript: str,
    inputdir: str,
    outputdir: str,
) -> batch_v1.Job:
    # Define what will be done as part of the job.
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = (
        f"{region}-docker.pkg.dev/{project_id}/containers/{container_name}:v1"
    )
    runnable.container.entrypoint = "/opt/conda/bin/python"
    runnable.container.commands = [f"/workdir/{runscript}", inputdir, outputdir]
    runnable.container.volumes = [f"/mnt/disks/{bucket_name}:/workdir"]

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
    resources.cpu_milli = 1000  # in milliseconds per cpu-second. This means the task requires 2 whole CPUs.
    resources.memory_mib = 1000  # in MiB
    task.compute_resource = resources

    task.max_retry_count = 2
    task.max_run_duration = "3600s"
    task.volumes = [vol]

    # Tasks are grouped inside a job using TaskGroups.
    # Currently, it's possible to have only one task group.
    group = batch_v1.TaskGroup()
    group.task_count = 3
    group.parallelism = 2
    group.task_spec = task

    # Policies are used to define on what kind of virtual machines the tasks will run on.
    # In this case, we tell the system to use "e2-standard-4" machine type.
    # Read more about machine types here: https://cloud.google.com/compute/docs/machine-types
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "e2-standard-4"
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
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
        for ff in glob(os.path.join(source, "**", "*"), recursive=True):
            relname = os.path.relpath(ff, source)
            target = f"{dest}/{relname}"
            print(f"Uploading {ff} to {target}")
            blob = bucket.blob(target)
            blob.upload_from_filename(ff)
    else:
        basename = os.path.basename(source)
        target = f"{dest}/{basename}"
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

    if source.endswith("/"):
        directory_mode = True
        files = _get_files_recursive(bucket, source, [])
    else:
        files = [source]

    logger.info(f"Downloading {files}")
    for ff in files:
        dest = target
        relname = os.path.relpath(ff, source)
        if directory_mode or os.path.isdir(target):
            dest = os.path.join(target, relname)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _download(bucket, ff, dest)


def inprogress(job_name):
    get_request = batch_v1.GetJobRequest(name=job_name)
    job = batch_client.get_job(request=get_request)
    if job.status.state == JobStatus.State.SUCCEEDED:
        print("Job succeeded")
        return 0
    elif job.status.state == JobStatus.State.FAILED:
        print("Job failed")
        return 0
    elif job.status.state in (
        JobStatus.State.QUEUED,
        JobStatus.State.RUNNING,
        JobStatus.State.SCHEDULED,
    ):
        print(f"Job status: {str(job.status.state)}")
    else:
        print(f"Unknown job status: {str(job.status.state)}")
    return 1


def wait(job_name, sleeptime=5, reporttime=None, reportcallback=None):
    import sys

    if reporttime is not None:
        if reporttime > sleeptime:
            from math import round

            reportfrequency = round(reporttime / sleeptime)
        else:
            reportfrequency = 1
            sleeptime = reporttime

    i = 1
    while True:
        inprog = inprogress(job_name)
        if reporttime is not None:
            if i == reportfrequency:
                if reportcallback is not None:
                    reportcallback(inprog)
                else:
                    logger.info(f"{inprog} jobs are pending completion")
                i = 1
            else:
                i += 1

        if inprog == 0:
            break

        sys.stdout.flush()
        time.sleep(sleeptime)


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
        self, bucket_name, job_name_prefix, runscript, inputdir, outputdir
    ):
        create_request = create_container_job(
            self.project,
            self.region,
            job_name_prefix,
            "moleculekit-service",
            bucket_name,
            runscript,
            inputdir,
            outputdir,
        )
        job = self.batch_client.create_job(create_request)
        return job.name

    def get_job_status(self, job_name):
        get_request = batch_v1.GetJobRequest(name=job_name)
        job = self.batch_client.get_job(request=get_request)
        return job.status.state

    def get_bucket(self, bucket_name):
        buckets = self.storage_client.list_buckets()
        bucket_names = [b.name for b in buckets]
        if bucket_name not in bucket_names:
            bucket = self.storage_client.create_bucket(bucket_name)
        else:
            bucket = self.storage_client.bucket(bucket_name)

        return bucket


class GoogleBatchJob(ProtocolInterface):
    def __init__(
        self,
        session: GoogleBatchSession,
        bucket_name: str,
        job_name_prefix: str = "job",
    ) -> None:
        super().__init__()
        self._session = session

        self._job_name = None
        self._job_name_prefix = job_name_prefix
        self._arg("inputpath", "string", "Input path", None, val.String())
        self._arg("remotepath", "string", "Remote path", None, val.String())
        self._arg("runscript", "string", "Run script", None, val.String())

        self._bucket = self._session.get_bucket(bucket_name)
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
        upload_to_bucket(self._bucket, self.inputpath, self.remotepath)
        self._job_name = self._session.create_container_job2(
            self._bucket_name,
            self._job_name_prefix,
            self.runscript,
            self.remotepath,
            self.remotepath + "/out",
        )

    def retrieve(self, path):
        remote = self.remotepath + "/out/"
        download_from_bucket(self._bucket, remote, path)
        # Cleaning up job
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
        _return_dataset=True,
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

        # if _return_dataset:
        #     try:
        #         result = self._datacenter.get_datasets(
        #             remotepath=f"{self._execid}/output", _logger=False
        #         )[0]
        #         return Dataset(
        #             datasetid=result["id"], _session=self._session, _props=result
        #         )
        #     except Exception:
        #         return None


if __name__ == "__main__":
    session = GoogleBatchSession(credential_file, project, "us-central1")
    job = GoogleBatchJob(session, "testbucket-moleculekit-1")
    job.inputpath = "./test/"
    job.remotepath = "test2"
    job.runscript = "test2/myscript.py"
    job.submit()
    job.wait()
    job.retrieve("./output/")
