from setuptools import setup, find_packages  

setup(  
    name="new_spectral_metric",  
    version="0.1.2",  
    packages=find_packages(),  
    install_requires=[  
        "torch",  
        "numpy",  
        "scipy",        
        "pandas",  
        "scikit-learn",   
        "matplotlib",           
    ],  
    author="CesarWKR",  
    author_email="cesarwkr1@gmail.com",  
    description="Descripción de tu paquete",  
    url="https://github.com/CesarWKR/CSG-Braycurtis.git",  
    classifiers=[  
        "Programming Language :: Python :: 3.10",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
    ],  
    python_requires='>=3.10',  
)  

