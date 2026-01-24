# 1. Base Image: Use Nginx (a super fast web server)
# "alpine" is a tiny Linux distribution (5MB size!)
FROM nginx:alpine

# 2. Configuration: Copy our custom nginx config (we will create this next)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 3. Copy Files: Move your HTML/CSS/JS into the folder Nginx serves from
COPY . /usr/share/nginx/html

# 4. Expose Port: Tell Docker this container listens on port 80
EXPOSE 80

# Nginx starts automatically, so no CMD needed