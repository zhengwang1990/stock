from setuptools import setup

setup(
    name='stock',
    version='1.0',
    packages=['stock'],
    url='https://github.com/zhengwang1990/stock',
    license='MIT',
    author='zhengwang',
    author_email='zheng.wang.rice@gmail.com',
    description='A stock trading strategy',
    install_requires=['tqdm', 'matplotlib', 'tabulate', 'yfinance', 'pandas', 'requests', 'numpy', 'retrying', 'sklearn', 'alpaca_trade_api',
                      'ta']
)
