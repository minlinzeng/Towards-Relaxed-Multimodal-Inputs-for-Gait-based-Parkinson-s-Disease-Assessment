# Source this file from ~/.bashrc to activate the project conda env when
# the shell enters this repository.

__cikm_auto_activate() {
  local project_dir="/home/yongjie/Minlin/CIKM-2025-Minlin"
  case "$PWD/" in
    "$project_dir"/*)
      if [ "${CONDA_DEFAULT_ENV:-}" != "minlin310" ]; then
        if [ -f "/home/yongjie/anaconda3/etc/profile.d/conda.sh" ]; then
          # shellcheck disable=SC1091
          . "/home/yongjie/anaconda3/etc/profile.d/conda.sh"
          conda activate minlin310 >/dev/null 2>&1
        fi
      fi
      ;;
  esac
}

case ";${PROMPT_COMMAND:-};" in
  *";__cikm_auto_activate;"*) ;;
  *) PROMPT_COMMAND="__cikm_auto_activate${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;;
esac

__cikm_auto_activate
