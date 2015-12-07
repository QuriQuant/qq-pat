from setuptools import setup

setup(name='qqpat',
      version='1.0',
      description='Python Financial Performance Analysys Tool',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Office/Business :: Financial :: Investment',
      ],
      keywords='funniest joke comedy flying circus',
      url='http://github.com/quriquant/qqpat',
      author='QuriQuant',
      author_email='suport@quriquant.com',
      license='MIT',
      packages=['qqpat'],
      install_requires=[
          'pandas',
          'seaborn'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )