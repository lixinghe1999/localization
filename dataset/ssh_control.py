import paramiko


def create_connection():
    # SSH connection details
    hostname = "192.168.137.172"
    username = "pi"
    password = "123456"

    # Create an SSH client
    client = paramiko.SSHClient()

    # Automatically add the host key (not recommended for production)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the SSH server
    client.connect(hostname=hostname, username=username, password=password)
    
    return client
def execute_command(client, command):
    # Execute the command
    # command = "ls -l"
    stdin, stdout, stderr = client.exec_command('cd ~/localization; ' + command)
    # Print the output
    print("Command output:")
    print(stdout.read().decode())

    # Print any error messages
    if stderr.read():
        print("Error output:")
        print(stderr.read().decode())
    return client
    # Close the SSH connection
    # client.close()
if __name__ == '__main__':
    client = create_connection()
    execute_command(client, 'ls -l')
