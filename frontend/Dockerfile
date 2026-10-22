# 多阶段构建 - 构建阶段
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖（包含devDependencies，构建时需要）
# NPM_REGISTRY 可在构建时覆盖，默认使用国内镜像以适配当前网络
ARG NPM_REGISTRY=https://repo.huaweicloud.com/repository/npm/
RUN npm config set registry ${NPM_REGISTRY} \
    && npm ci --no-audit --no-fund --fetch-retries=8 --fetch-timeout=600000

# 复制配置文件
COPY next.config.js ./
COPY tsconfig.json ./
COPY tailwind.config.js ./
COPY postcss.config.js ./

# 复制源代码
COPY src ./src

# 构建应用
RUN npm run build

# 生产阶段
FROM node:18-alpine AS runner

# 设置工作目录
WORKDIR /app

# 创建非root用户
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# 复制构建产物（无 public 静态资源；如后续新增，需恢复 public 的 COPY）
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# 设置文件权限
RUN chown -R nextjs:nodejs /app

# 切换到非root用户
USER nextjs

# 暴露端口
EXPOSE 3000

# 设置环境变量
ENV PORT 3000
ENV NODE_ENV production

# 启动应用
CMD ["node", "server.js"]