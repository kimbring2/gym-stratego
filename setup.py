from distutils.core import setup

setup(
  name = 'gym_stratego',         
  packages = ['gym_stratego'],   
  version = '0.0.1',      
  license='MIT',        
  description = 'An OpenAI Gym for Stratego board game to benchmark Reinforcement Learning algorithms',   
  author = 'kimbring2',                   
  author_email = 'kimbring2@gmail.com',      
  url = 'https://github.com/kimbring2/gym-stratego',   
  download_url = 'https://github.com/kimbring2/gym-stratego/',    
  keywords = ['Machine Learning', 'Reinforcement Learning', 'Board Game'],   
  install_requires=[            
          'numpy',
          'gym',
          
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Science/Research',      
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.7',
  ],
)