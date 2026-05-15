FROM golang:1.21-alpine AS builder
WORKDIR /app
# Note: Requires go mod init asp_cluster && protoc generation prior to build
COPY . .
RUN go build -o orbit_dispatch cmd/orbit_dispatch/main.go

FROM alpine:latest
WORKDIR /root/
COPY --from=builder /app/orbit_dispatch .
EXPOSE 50051
CMD ["./orbit_dispatch"]