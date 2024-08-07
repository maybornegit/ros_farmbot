from setuptools import setup
import os
from glob import glob

package_name = 'ros_farmbot'
scripts = 'ros_farmbot/scripts'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, scripts],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='frc-ag-2',
    maintainer_email='morgan8211@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'init = ros_farmbot.central_i2grow_automation:main',
        	'ui = ros_farmbot.webcam_test:main',
        	'listen = ros_farmbot.data_listener:main',
        ],
    },
)
