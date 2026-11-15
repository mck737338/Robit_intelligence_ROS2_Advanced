from setuptools import setup

package_name = 'arm_ik'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jammay',
    maintainer_email='jammay0820@gmail.com',
    description='DH 기반 6축 IK 계산 및 robot_description 비교 검증',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_ik_verify = arm_ik.arm_ik_verify_node:main',
        ],
    },
)
