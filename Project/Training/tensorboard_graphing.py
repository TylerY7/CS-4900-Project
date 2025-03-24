import os
import webbrowser
from subprocess import Popen, DEVNULL
import time

# update this later to make sure script can run from any directory and find the correct path
log_dir = os.path.join(os.getcwd(), 'runs')

# make sure path can be found, otherwise exit
if not os.path.exists(log_dir):
    print(f'ERROR: Log director \'{log_dir}\' not found.')
    exit(1)

# opens TB process
print('Starting TensorBoard...')
# stdout and stderr redirected as to not fill terminal with text
tb_process = Popen(['tensorboard', '--logdir', log_dir], stdout=DEVNULL, stderr=DEVNULL)

# wait before opening web server so that it has time to start
time.sleep(5)

# opens web browser to default URL used by tensorboard
url = 'http://127.0.0.1:6006'
print(f'Opening TensorBoard at {url}')
webbrowser.open(url)

print(f'TensorBoard is running at {url}\nPress CTRL+C to stop.')
try:
    tb_process.wait()
except KeyboardInterrupt:
    print('\nShutting down TensorBoard...')
    tb_process.terminate()