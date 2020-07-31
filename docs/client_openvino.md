# client_openvino.py

The client_openvino program is a client inference program compatible with the distributed hosts, designed to run on any
device compatible with Intel's OpenVINO software. 

## Arguments

`--model (-m)` - File path of the detection model. This is required. Note that this should specify the XML file, and the
binary file should be of the same name. 

`--ip` - IP of the host (IPv4), including port (`ip:port`). If port is not specified, port 8080 is assumed. This is required.

`-d` - Device to be used. Accepts CPU, GPU, FPGA, HDDL, or MYRIAD. CPU by default. 

`-l` - Optional argument for OpenVINO custom CPU layers. 

`--verbose, -v` - Enables verbose logging. 

`--report_interval (-r)` - Report interval for inference metrics, in seconds. Default is 10. 

##Usage

To use, simply run the python file AFTER the host has been enabled.