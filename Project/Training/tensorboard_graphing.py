import os
import webbrowser
from subprocess import Popen, DEVNULL
import time

log_dir = os.path.join(os.getcwd(), 'runs')

if not os.path.exists(log_dir):
    print(f'ERROR: Log director \'{log_dir}\' not found.')
    exit(1)

print('Starting TensorBoard...')
tb_process = Popen(['tensorboard', '--logdir', log_dir], stdout=DEVNULL, stderr=DEVNULL)

time.sleep(5)

url = 'http://localhost:6006'
print(f'Opening TensorBoard at {url}')
webbrowser.open(url)

print('TensorBoard is running at http://localhost:6006\nPress CTRL+C to stop.')
try:
    tb_process.wait()
except KeyboardInterrupt:
    print('\nShutting down TensorBoard...')
    tb_process.terminate()