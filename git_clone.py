import os
from getpass import getpass
import urllib

user = input('User name: ')
password = getpass('Password: ')
password = urllib.parse.quote(password) # your password is converted into url format
repo_name = input('Repo name: ')

cmd_string = 'git clone https://{0}:{1}@github.com/{0}/{2}.git'.format(user, password, repo_name)

os.system(cmd_string)
cmd_string, password = "", "" # removing the password from the variable