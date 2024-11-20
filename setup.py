from setuptools import setup, find_packages

setup(
  name='hendraix',
  version='0.1.0',
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  install_requires=[
    # e.g., 'numpy', 'requests',
  ],
  entry_points={
    'console_scripts': [
        'my_command = hendraix.main:main',
    ],
  },
  author='Sean Kim',
  author_email='seandkim14@gmail.com',
  description='A brief description of your project',
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  url='https://github.com/yourusername/hendraix',
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: Mac OS',
  ],
  python_requires='>=3.12',
)