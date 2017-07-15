import names
import random
import os

os.chdir('/Users/JAustin/Documents/C++')
with open('example_short_space.txt', 'w+') as file:
    for i in range(100):
        file.write('%s %d %d %d %d %d %d \n' %(names.get_first_name(), 100*random.random(), 100*random.random(), 100*random.random(), 100*random.random(), 100*random.random(), 100*random.random()))
        # file.write('%s \n' %names.get_first_name())
