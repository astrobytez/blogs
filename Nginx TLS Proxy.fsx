(**
# Guide to deploy WebApp with TLS enabled behind an NGINX proxy

This article covers my experience and dealing with gotchas setting up a web server behind a Nginx Proxy with TLS support enabled.
I hope to cover the main pain points I have experienced to help others get going.

## 1 - Configure a VPS (Virtual Private Server)

First off I'll assume you have a Virtual Private Server with SSH access configured. If not you can get started with the following link at
[digital ocean](https://docs.digitalocean.com/products/droplets/how-to/create/).

I would also recommend configuring a new user with sudo permission (add to ```wheel``` group) and revoking root SSH access for added security.
With a VPS configured and running and a SSH connection established we can setup the prerequisites. This article will assume the VPS is a Fedora based system ;).
You can follow [this](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-20-04) guide to setup the new user and [this](https://www.digitalocean.com/community/tutorials/how-to-disable-root-login-on-ubuntu-20-04)
guide to remove root access.


Firstly ensure the system is up to date: ```sudo dnf -y update && reboot now```
Login once again once the reboot is complete.

```bash
# Install the dependencies
sudo dnf -y install nginx certbot python3-certbot-nginx

# Check nginx status
systemctl status nginx
```
Before you can recieve incoming requests you will have to ensure the firewall for the server has both ports 80 and 443 open for incoming traffic.

Next you can enable nginx and start the server.
```bash
sudo systemctl enable nginx.service
sudo systemctl start nginx.service
```

Now you can open a browser and with the IP address of the VPS server you can go to the landing page of nginx to confirm its running:
Navigate to: ```http://your-vps-public-ip-addr```

If you navigate to the server address and see the Nginx landing page, congratulations you have an Nginx server running.

## 2 - Configure Domain Names and Server Blocks

Next we will configure the server to use TLS encryption using the [Lets Encrypt](https://letsencrypt.org/) certificate bot [Certbot](https://certbot.eff.org/).
In order to use TLS encryption you have to have control of the server running at the location the DNS entry points to.

Therefore you will need to own a domain in order to configure TLS for use in a production system. You can get domain names setup easily with [Couldflare Registrar](https://www.cloudflare.com/products/registrar/)

At this point you must own a domain name and then setup the DNS class A entries to point to the IP address of the server.
Follow the steps provided by the domain name service provider on how to configure your domain name records.

With your DNS entries setup and pointing to the public IP of the server we can now configure TLS using Certbot.

### Server blocks
Next you will setup Nginx server blocks.
This allows Nginx to run multiple virtual servers and route incoming requests to multiple endpoints behind the proxy.

Refer [here](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-20-04#step-5-setting-up-server-blocks-recommended) for more information.

```bash
sudo mkdir -p /etc/nginx/sites-{available,enabled}
sudo vi /etc/nginx/sites-available/your_domain # use nano if you prefer
```


Edit ```/etc/nginx/sites-available/your_domain``` for editing, i.e.; ```sudo vi /etc/nginx/sites-available/your_domain```.

```text
server {
        listen 80;
        listen [::]:80;

        server_name your_domain www.your_domain;

        location / {
                # https://www.digitalocean.com/community/tutorials/how-to-configure-nginx-as-a-reverse-proxy-on-ubuntu-22-04#step-2-configuring-your-server-block
                proxy_pass web_app_address # For example: http://localhost:8080
                proxy_set_header Host $http_host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;

                # WebSocket support (uncomment if in use) - https://stackoverflow.com/a/39216976
                # proxy_http_version 1.1;
                # proxy_set_header Upgrade $http_upgrade;
                # proxy_set_header Connection "upgrade";
        }
}
```

With the server block in sites available we now must link it to sites enabled.
```bash
sudo ln -s /etc/nginx/sites-available/your_domain /etc/nginx/sites-enabled/
```

Add sites-enabled to the main nginx configuration:
Edit ```/etc/nginx/nginx/conf``` for editing, i.e.; ```sudo vi /etc/nginx/nginx/conf```.

```bash
sudo vi /etc/nginx/nginx.conf
# look for the line:
       include /etc/nginx/conf.d/*.conf # within the http block.
# add the following line after:
       include /etc/nginx/sites-enabled/*.* # ***NOTE: the double wildcard here is [important](https://stackoverflow.com/a/41452450):***
```

Restart Nginx
```bash
sudo systemctl restart nginx
```

At this point you will want to run the Web Application of your choosing and ensure its listening at the address you provided in the proxy configuration.
If you want a simple test program you can do the following:

```bash
mkdir dummy-app
echo "Hello World!!!" >> dummy-app/index.html
python -m http.server -d dummy-app 8080 # you can now use http://localhost:8080 as the proxy_pass address above.
```

You should now be able to navigate to your domain and see "Hello World!!!" or the content of your application running.
To enable TLS support and site encryption the next step is to run CertBot to generate certificate keys and install them in Nginx.
If you encounter a 403 Forbidden error at this point this is likely because of SElinux policy defaults.

The solution I found to this can be found [here](https://stackoverflow.com/a/26228135).

## 3 - Install TLS certificates

```bash
# Run certbot to generate and install the certificate keys. Follow the prompts.
# Ensure all the domains you provide here have associated DNS A entries created.
sudo certbot --nginx -d your_domain -d www.your_domain

# Example: certbot --nginx -d example.com -d www.example.com
```

If everything is successful you should now have TLS support. The server should now only support HTTPS connections. Certbot automatically
configures the server configuration to only accept HTTPS connections and will redirect any HTTP connections. It will also setup a scheduled
service to renew the certificate keys automatically.

I hope this helps and thanks for reading!

*)