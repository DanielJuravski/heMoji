## run the following cmd to build a new docker heMoji img
docker build . -t danieljuravski/hemoji:1.0.$(git rev-parse --short HEAD) -t danieljuravski/hemoji:latest

