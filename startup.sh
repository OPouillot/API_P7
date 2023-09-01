RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
