# client_edgetpu.py

The client_edgetpu program is a client inference program compatible with the distributed hosts, designed to run on any
device compatible with Google's EdgeTPU software. 

## Arguments

`--model` - File path of the detection model. This is required.

`--ip` - IP of the host (IPv4), including port (`ip:port`). If port is not specified, port 8080 is assumed. This is required.

`--verbose, -v` - Enables verbose logging. 

##Usage

To use, simply run the python file AFTER the host has been enabled.