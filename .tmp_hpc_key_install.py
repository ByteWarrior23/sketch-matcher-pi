import os
import paramiko

host, user = "10.10.11.201", "ee_24126016"
password = os.environ["HPC_PASSWORD"]
pub = open(os.path.expanduser("~/.ssh/id_ed25519_hpc.pub")).read().strip()
key_fp = pub.split()[1]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=22, username=user, password=password, timeout=30)

def run(cmd):
    _in, out, err = c.exec_command(cmd)
    return out.channel.recv_exit_status(), out.read().decode(), err.read().decode()

rc, out, err = run("grep -qF '%s' ~/.ssh/authorized_keys 2>/dev/null && echo INSTALLED || echo MISSING" % key_fp)
if "INSTALLED" in out:
    print("Key already installed.")
else:
    rc, out, err = run("mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '%s' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED" % pub)
    print("install rc=%d out=%s err=%s" % (rc, out.strip(), err.strip()))

rc, out, err = run("hostname; whoami; echo ---; nvidia-smi -L 2>/dev/null | wc -l; echo ---; ls -d /Data 2>/dev/null; echo ---; python3 --version 2>/dev/null")
print("--- host snapshot ---")
print(out)
print("err:", err.strip() if err.strip() else "(none)")
c.close()
