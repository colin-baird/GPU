FROM ubuntu:24.04

ARG TZ=America/Los_Angeles
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install C++/CMake toolchain, RISC-V cross-compiler, and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
  # C++ build toolchain
  build-essential \
  cmake \
  ninja-build \
  gdb \
  # RISC-V cross-compilation
  gcc-riscv64-unknown-elf \
  # Networking/firewall (required for init-firewall.sh)
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  curl \
  ca-certificates \
  # Dev tools
  git \
  less \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  jq \
  nano \
  vim \
  # Node.js (for Claude Code CLI)
  nodejs \
  npm \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME && \
  useradd --uid $USER_UID --gid $USER_GID -m -s /bin/zsh $USERNAME && \
  echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME

# Persist bash/zsh history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Set environment
ENV DEVCONTAINER=true

# Create workspace and config directories
RUN mkdir -p /workspace /home/$USERNAME/.claude && \
  chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude

WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Set the default shell and editor
ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Install zsh with powerline10k
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(curl -fsSL https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude Code CLI
ENV NPM_CONFIG_PREFIX=/home/$USERNAME/.npm-global
ENV PATH=$PATH:/home/$USERNAME/.npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "$USERNAME ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/$USERNAME-firewall && \
  chmod 0440 /etc/sudoers.d/$USERNAME-firewall
USER $USERNAME
