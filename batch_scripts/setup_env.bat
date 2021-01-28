call cd ..
call conda env create -f environment.yml
call conda activate batopt
call ipython kernel install --user --name=batopt
pause