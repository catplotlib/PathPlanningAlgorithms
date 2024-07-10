from setuptools import find_packages, setup

package_name = 'reach'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
     install_requires=['setuptools','qiskit'],
    zip_safe=True,
    maintainer='puja',
    maintainer_email='puja@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar_node = reach.astar_node:main',
            'pp_node= reach.pp_node:main',
            'occupancy_grid_node = reach.occupancy_grid_node:main',
        ],
    },
)
