# Distributed Inference Platform

The distributed inference platform is a set of tools for distributing inference tasks to a heterogeneous network of EDGE devices. 
The programs in this repository allow for sending an image or video stream to an EDGE inference device, receive an 
inference result, and overlay the inference result onto the image or video stream. Currently, the EDGE devices supported
are those running on Google's EDGETPU platform, Nvidia's TensorRT platform, or Intel's OpenVINO platform. Network
transmission can occur locally or over the internet. Each server and client tool runs with a number of metrics that can
be used to empirically analyze network inference and distribution.  

## Requirements

See the [requirements file](REQUIREMENTS.md) for details on required packages and installations.

## Contents

This section details a brief summary of available programs. For in-depth, documentation, please consult the *docs* folder.

`host.py` - Class for distributed inference host, and single-image distributed inference test

`stream.py` - Class for distributed inference video streaming

`relay.py` - Class for client-side network interface

`client_edgetpu.py`, `client_tensorrt.py`, `client_openvino.py` - Client inference programs for respective devices

`di_utils.py` - Various distributed inference utilities

`stopwatch.py` - Class for collecting data on duration of image operations on EDGE devices