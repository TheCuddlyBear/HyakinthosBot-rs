
FROM rust:1 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       cmake \
       build-essential \
       pkg-config \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
       python3 \
       curl \
       unzip \
    && rm -rf /var/lib/apt/lists/*

# Install the latest standalone yt-dlp release.
RUN curl -fsSL \
    https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /usr/local/bin/yt-dlp \
    && chmod 755 /usr/local/bin/yt-dlp \
    && yt-dlp --version

# Install Deno for YouTube JavaScript challenge support.
RUN curl -fsSL https://deno.land/install.sh -o /tmp/install-deno.sh \
    && DENO_INSTALL=/usr/local sh /tmp/install-deno.sh \
    && rm /tmp/install-deno.sh

WORKDIR /app

COPY --from=builder /app/target/release/HyakinthosBot /usr/local/bin/HyakinthosBot
COPY config.example.yml ./config.yml

CMD ["/usr/local/bin/HyakinthosBot"]