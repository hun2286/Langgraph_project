# 1단계: 빌드 (Vite 정적 파일 생성)
FROM node:18-alpine AS build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 2단계: Nginx로 서빙 (매우 가볍고 빠름)
FROM nginx:stable-alpine
# Vite 빌드 결과물(dist)을 Nginx의 기본 경로로 복사
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]