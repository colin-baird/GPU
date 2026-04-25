FROM ubuntu:24.04

ARG TZ=America/Los_Angeles
ENV TZ="$TZ"
ARG CLAUDE_CODE_VERSION=latest
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# --- Base packages (from ~/devcontainer-template/Dockerfile.base) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  git \
  less \
  procps \
  sudo \
  fzf \
  zsh \
  unzip \
  jq \
  nano \
  nodejs \
  npm \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Project-specific packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  gdb \
  gcc-riscv64-unknown-elf \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- User setup (from base template) ---
RUN existing_user=$(getent passwd $USER_UID | cut -d: -f1) && \
  if [ -n "$existing_user" ] && [ "$existing_user" != "$USERNAME" ]; then \
    usermod -l $USERNAME -d /home/$USERNAME -m "$existing_user" && \
    groupmod -n $USERNAME $(getent group $USER_GID | cut -d: -f1) 2>/dev/null || true; \
  elif [ -z "$existing_user" ]; then \
    groupadd --gid $USER_GID $USERNAME 2>/dev/null || true && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
  fi && \
  chsh -s /bin/zsh $USERNAME && \
  echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir /commandhistory && touch /commandhistory/.bash_history && \
  chown -R $USERNAME /commandhistory

RUN mkdir -p /Users/colinbaird && \
  ln -s /home/$USERNAME/.claude /Users/colinbaird/.claude

ENV DEVCONTAINER=true

RUN mkdir -p /workspace /home/$USERNAME/.claude && \
  chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME/.claude

WORKDIR /workspace
USER $USERNAME

ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
  sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="clean"/' /home/$USERNAME/.zshrc && \
  echo 'export HISTFILE=/commandhistory/.bash_history' >> /home/$USERNAME/.zshrc && \
  echo "alias dangerclaude='claude --dangerously-skip-permissions'" >> /home/$USERNAME/.zshrc

ENV NPM_CONFIG_PREFIX=/home/$USERNAME/.npm-global
ENV PATH=$PATH:/home/$USERNAME/.npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
