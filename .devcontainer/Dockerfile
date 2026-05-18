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
# graphviz: `dot`, used by tools/render_signal_diagram.py --svg.
# clang/libclang/python3-clang: version-matched libclang library, compiler
# resource headers, and Python bindings for the AST signal-diagram extractor
# (tools/diagram_extract_ast.py, used by render_signal_diagram.py --validate)
# and the timing-naming lint's cross-module-read analysis pass. The explicit
# CLANG_RESOURCE_DIR avoids partial ASTs when libclang cannot find builtin
# compiler headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  clang-18 \
  cmake \
  ninja-build \
  gdb \
  gcc-riscv64-unknown-elf \
  graphviz \
  libclang-18-dev \
  python3-clang-18 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LIBCLANG_LIBRARY_PATH=/usr/lib/llvm-18/lib
ENV CLANG_RESOURCE_DIR=/usr/lib/llvm-18/lib/clang/18

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
  echo "alias dangerclaude='claude --dangerously-skip-permissions'" >> /home/$USERNAME/.zshrc && \
  echo '{"skipDangerousModePermissionPrompt": true}' > /home/$USERNAME/.claude/settings.local.json

ENV NPM_CONFIG_PREFIX=/home/$USERNAME/.npm-global
ENV PATH=$PATH:/home/$USERNAME/.npm-global/bin
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
