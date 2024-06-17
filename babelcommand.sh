#/bin/bash!
podman --log-level=debug run --name babelnet --rm -it -d -v /ceph/tgreen/resources/BabelNet-5.3/:/root/babelnet -p 6565:1234 babelscape/babelnet-rpc:latest