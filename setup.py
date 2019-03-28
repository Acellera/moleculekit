import setuptools
import unittest

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir='moleculekit', pattern='*.py')
    return test_suite

if __name__ == '__main__':
    with open('README.md', 'r') as fh:
        long_description = fh.read()

    setuptools.setup(
        name='moleculekit',
        version='HTMDMOL_VERSION_PLACEHOLDER',
        author='Acellera',
        author_email='info@acellera.com',
        description='A molecule reading/writing and manipulation package.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/acellera/moleculekit/',
        classifiers=[
            'Programming Language :: Python :: PYTHON_VERSION_PLACEHOLDER',
            'Operating System :: POSIX :: Linux',
        ],

        packages=setuptools.find_packages(include=['moleculekit*'], exclude=['test-data', '*test*', ]),

        package_data={
            'moleculekit': ['lib/*/*', 'vmd_wrapper'],
        },

        test_suite='setup.my_test_suite',
        install_requires=[
            DEPENDENCY_PLACEHOLDER
        ]
    )