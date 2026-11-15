# Frontend image (Phase 6, Step 33): the built UI served by nginx. The API addresses are fixed at
# build time (the browser calls the published localhost ports directly).
FROM node:22-alpine AS build
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ARG VITE_API_URL=http://127.0.0.1:28100
ARG VITE_SERVING_URL=http://127.0.0.1:28200
ENV VITE_API_URL=$VITE_API_URL VITE_SERVING_URL=$VITE_SERVING_URL
RUN npm run build

FROM nginx:stable-alpine
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /ui/dist /usr/share/nginx/html
USER nginx
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
