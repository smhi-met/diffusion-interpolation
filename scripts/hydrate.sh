# Resolve Hydra-like interpolations in a YAML file and write a fully-resolved copy
# into the resolved _work_dir/config.yaml (keeping ALL other config lines).
#
# Supports:
#   ${now:<strftime>}          -> date +"<strftime>"
#   ${oc.env:VAR}              -> environment variable VAR
#   ${oc.env:VAR,default}      -> environment variable VAR or default
#   ${_experiment_name} etc.   -> recursive key lookup inside the YAML (top-level keys)
#
# Usage:
#   WORK_DIR="$(hydrate_config experiments/example/atos/check-configs/some-configs.yaml)" || exit 1
#   echo "Resolved config written to: $WORK_DIR/config.yaml"

hydrate_config() {
    local yaml_file="$1"
    local out_name="${2:-config.yaml}"

    if [ -z "$yaml_file" ] || [ ! -f "$yaml_file" ]; then
        echo "Error: YAML file not found: '$yaml_file'" >&2
        return 1
    fi

    local FIXED_NOW_STAMP
    FIXED_NOW_STAMP=$(date +"%s")


    # ---------- helpers ----------
    _trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

    _strip_quotes() {
        local s="$1"
        s="${s#\"}"; s="${s%\"}"
        s="${s#\'}"; s="${s%\'}"
        printf '%s' "$s"
    }

    # Extract first "key:" occurrence from a file (defaults to yaml_file), return raw value string
    _get_key_raw() {
        local key="$1"
        local file="${2:-$yaml_file}"
        local line
        line="$(grep -E "^[[:space:]]*${key}:" "$file" | head -n 1)"
        [ -n "$line" ] || return 1
        local val
        val="$(echo "$line" | cut -d: -f2- | _trim)"
        _strip_quotes "$val"
    }

    # Resolve all expressions in a string, repeatedly until stable
    _resolve_exprs() {
        local s="$1"
        local depth="${HYDRA_PARSE_DEPTH:-0}"
        if (( depth > 30 )); then
            echo "Error: recursion too deep while resolving expressions." >&2
            return 1
        fi

        local changed=1
        while (( changed )); do
            changed=0

            # ${now:...}
            while [[ "$s" =~ \$\{now:([^}]+)\} ]]; do
                local fmt="${BASH_REMATCH[1]}"
                local ts
                ts="$(date -d "@$FIXED_NOW_STAMP" +"$fmt")"
                s="${s//\$\{now:$fmt\}/$ts}"
                changed=1
            done

            # ${oc.env:VAR} and ${oc.env:VAR,default}
            while [[ "$s" =~ \$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(,([^}]*))?\} ]]; do
                local var="${BASH_REMATCH[1]}"
                local def="${BASH_REMATCH[3]}"
                local repl="${!var}"
                if [ -z "$repl" ] && [ -n "$def" ]; then
                    repl="$def"
                fi
                local full="${BASH_REMATCH[0]}"
                s="${s//$full/$repl}"
                changed=1
            done

            # ${some_key} (recursive YAML key reference)
            while [[ "$s" =~ \$\{([A-Za-z0-9_]+)\} ]]; do
                local ref="${BASH_REMATCH[1]}"
                local full="${BASH_REMATCH[0]}"

                # avoid self-loop at string level
                if [ "$ref" = "_RESOLVE_SELF_" ]; then
                    echo "Error: invalid self reference." >&2
                    return 1
                fi

                local ref_raw
                if ! ref_raw="$(_get_key_raw "$ref")"; then
                    echo "Error: referenced key '$ref' not found while resolving '$s'." >&2
                    return 1
                fi

                local ref_val
                ref_val="$(HYDRA_PARSE_DEPTH=$((depth+1)) _resolve_exprs "$ref_raw")" || return 1

                s="${s//$full/$ref_val}"
                changed=1
            done
        done

        printf '%s' "$s"
    }

    # Replace only the first occurrence of each key line in-place within a file
    _replace_first_key_line() {
        local file="$1"
        local key="$2"
        local value="$3"

        # escape for sed replacement (delimiter | and &)
        local esc
        esc="$(printf '%s' "$value" | sed -e 's/[&|]/\\&/g')"

        # GNU sed: 0,/pattern/ means first match only
        sed -i \
          -e "0,/^[[:space:]]*${key}[[:space:]]*:/s|^[[:space:]]*${key}[[:space:]]*:.*|${key}: ${esc}|" \
          "$file"
    }

    # ---------- resolve base keys ----------
    local raw_exp raw_run raw_work
    raw_exp="$(_get_key_raw "_experiment_name")" || { echo "Error: missing _experiment_name" >&2; return 1; }
    raw_run="$(_get_key_raw "_run_name")"        || { echo "Error: missing _run_name" >&2; return 1; }
    raw_work="$(_get_key_raw "_work_dir")"       || { echo "Error: missing _work_dir" >&2; return 1; }

    local experiment_name run_name work_dir
    experiment_name="$(_resolve_exprs "$raw_exp")" || return 1
    run_name="$(_resolve_exprs "$raw_run")"        || return 1
    work_dir="$(_resolve_exprs "$raw_work")"       || return 1

    [ -n "$work_dir" ] || { echo "Error: resolved _work_dir is empty" >&2; return 1; }
    mkdir -p "$work_dir" || return 1

    local out_path="$work_dir/$out_name"
    cp -f "$yaml_file" "$out_path" || return 1

    # Fill the first occurrences of the base keys
    _replace_first_key_line "$out_path" "_experiment_name" "$experiment_name" || return 1
    _replace_first_key_line "$out_path" "_run_name"        "$run_name"        || return 1
    _replace_first_key_line "$out_path" "_work_dir"        "$work_dir"        || return 1

    # ---------- resolve any remaining ${...} anywhere in the file ----------
    # We do a few passes until stable.
    local i=0
    while (( i < 10 )); do
        if ! grep -q '\${' "$out_path"; then
            break
        fi

        # Read/resolve line-by-line to preserve file layout as much as possible
        local tmp="${out_path}.tmp"
        : > "$tmp" || return 1

        while IFS= read -r line || [ -n "$line" ]; do
            if [[ "$line" == *'${'* ]]; then
                # Resolve expressions only in the value part; keep the original key/indent if present
                # This is conservative: resolves interpolations anywhere on the line.
                local resolved
                resolved="$(_resolve_exprs "$line")" || return 1
                printf '%s\n' "$resolved" >> "$tmp" || return 1
            else
                printf '%s\n' "$line" >> "$tmp" || return 1
            fi
        done < "$out_path"

        mv "$tmp" "$out_path" || return 1
        ((i++))
    done
    # Read resolved values from out_path so any interpolations are already expanded
    local dataset_path assets_dir
    dataset_path="$(_get_key_raw "_dataset_path" "$out_path")" || { echo "Error: missing _dataset_path" >&2; return 1; }
    assets_dir="$(_get_key_raw "_assets_dir" "$out_path")" || assets_dir=""

    echo "$work_dir" "$dataset_path" "$assets_dir"
}
