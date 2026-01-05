from setuptools import setup
import os
from glob import glob

package_name = 'gesture_emotion_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # Registro en ROS 2
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'mediapipe',
        'fer',
        'tensorflow',
        'mtcnn'
    ],
    zip_safe=True,
    maintainer='javier',
    maintainer_email='javier@todo.com',
    description='Nodo de detección de gestos y expresiones faciales',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Aquí se conecta ros2 run gesture_emotion_ros gesture_emotion_node
            'gesture_emotion_node = gesture_emotion_ros.gesture_emotion_node:main',
        ],
    },
)
