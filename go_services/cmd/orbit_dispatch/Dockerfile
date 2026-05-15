FROM golang:1.21-alpine AS builder
RUN apk add --no-cache protobuf protobuf-dev curl
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31.0 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3.0

WORKDIR /app
COPY go_services/go.mod go_services/go.sum ./
RUN go mod download && go mod verify

COPY go_services/proto ./proto
RUN mkdir -p pb && \
    protoc --go_out=pb --go_opt=paths=source_relative \
           --go-grpc_out=pb --go-grpc_opt=paths=source_relative \
           -I proto proto/asp_cluster.proto

COPY go_services/cmd ./cmd
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o orbit_dispatch ./cmd/orbit_dispatch

FROM scratch
COPY --from=builder /app/orbit_dispatch /orbit_dispatch
EXPOSE 50051
ENTRYPOINT ["/orbit_dispatch"]