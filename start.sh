cd ~/deploy
ip=`ifconfig -a | grep 'inet' | grep -v '127.0.0.1' | grep 'inet6' -v | awk '{print $2}' | tr -d "addr:"`
echo "starting the server ip : $ip"
python3 server.py --ip $ip --port 2022