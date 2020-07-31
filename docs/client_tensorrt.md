# client_tensorrt.py

The client_tensorrt program is a client inference program compatible with the distributed hosts, designed to run on any
device compatible with Nvidia's TensorRT software. 

## Arguments

`--model (-m)` - File path of the detection model. This is required. This should be a serialized binary file compiled
for a CUDA machine. 

`--ip` - IP of the host (IPv4), including port (`ip:port`). If port is not specified, port 8080 is assumed. This is required.

`--verbose, -v` - Enables verbose logging. 

`--report_interval (-r)` - Report interval for inference metrics, in seconds. Default is 10. 

##Usage

To use, simply run the python file AFTER the host has been enabled.