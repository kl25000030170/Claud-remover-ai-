# Use a base image with both Python 3.11 and Node.js 20 pre-installed
FROM nikolaik/python-nodejs:python3.11-nodejs20-slim

# Install system libraries needed by OpenCV and compile tools
# Replaced deprecated libgl1-mesa-glx with non-deprecated graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy Node.js dependency manifests
COPY package.json package-lock.json ./

# Install Node.js production & build dependencies
RUN npm ci

# Set up Python virtual environment
RUN python -m venv .venv
RUN .venv/bin/pip install --no-cache-dir --upgrade pip

# Copy Python requirements and install CPU-optimized packages
COPY requirements.txt ./
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace files
COPY . .

# Compile and build the TanStack Start production app for Node.js target preset
ENV NITRO_PRESET=node-server
ENV NODE_ENV=production
RUN npm run build

# Expose default port (Railway will override $PORT dynamically)
ENV PORT=8080
EXPOSE 8080

# Configure startup command
RUN chmod +x start.sh
CMD ["./start.sh"]
