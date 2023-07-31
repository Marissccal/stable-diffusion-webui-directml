# this scripts installs necessary requirements and launches main program in webui.py
import re
import subprocess
import os
import sys
import importlib.util
import platform
import json
import shutil
from functools import lru_cache

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir
from modules import timer

timer.startup_timer.record("start")

args, _ = cmd_args.parser.parse_known_args()

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"

# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

if 'GRADIO_ANALYTICS_ENABLED' not in os.environ:
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        import modules.errors

        modules.errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3106/

{"Alternatively, use a binary release of WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases" if is_windows else ""}

Use --skip-python-version-check to suppress this warning.
""")


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output([git, "rev-parse", "HEAD"], shell=False, encoding='utf8').strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag():
    try:
        return subprocess.check_output([git, "describe", "--tags"], shell=False, encoding='utf8').strip()
    except Exception:
        try:

            changelog_md = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CHANGELOG.md")
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((line.strip() for line in file if line.strip()), "<none>")
                line = line.replace("## ", "")
                return line
        except Exception:
            return "<none>"


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def run_pip_uninstall(package: str, desc: str = None):
    if args.skip_install:
        return

    return run(f'"{python}" -m pip uninstall {package}', desc=f"Uninstalling {desc}", errdesc=f"Couldn't uninstall {desc}")


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    return result.returncode == 0


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C "{dir}" rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C "{dir}" fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C "{dir}" checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def version_check(commit):
    try:
        import requests
        commits = requests.get('https://api.github.com/repos/lshqqytiger/stable-diffusion-webui-directml/branches/master').json()
        if commit != "<none>" and commits['commit']['sha'] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print("version check failed", e)


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.path.abspath('.')}{os.pathsep}{env.get('PYTHONPATH', '')}"

        print(run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env))
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        if os.path.isfile(settings_file):
            with open(settings_file, "r", encoding="utf8") as file:
                settings = json.load(file)
    except Exception:
        errors.report("Could not load settings", exc_info=True)

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if disable_all_extensions != 'none':
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(extensions_dir):
        return

    for dirname_extension in list_extensions(settings_file):
        run_extension_installer(os.path.join(extensions_dir, dirname_extension))


re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """

    import importlib.metadata
    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(re_requirement, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    if args.backend == 'auto':
        nvidia_driver_found = shutil.which('nvidia-smi') is not None
        rocm_found = shutil.which('rocminfo') is not None
        if nvidia_driver_found:
            args.backend = 'cuda'
            print("NVIDIA driver was found. Automatically changed backend to 'cuda'. You can manually select which backend will be used through '--backend' argument.")
        elif rocm_found:
            args.backend = 'rocm'
            print("ROCm was found. Automatically changed backend to 'rocm'. You can manually select which backend will be used through '--backend' argument.")
        else:
            args.backend = 'directml'
        cmd_args.parser.set_defaults(backend=args.backend)
    if args.backend == 'cuda':
        torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    if args.backend == 'rocm':
        torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/rocm5.4.2")
    if args.backend == 'directml':
        torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==2.0.0 torchvision==0.15.1 torch-directml")
    
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    requirements_file_onnx = os.environ.get('REQS_FILE', "requirements_onnx.txt")
    requirements_file_olive = os.environ.get('REQS_FILE', "requirements_olive.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')
    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "https://github.com/TencentARC/GFPGAN/archive/8d2447a2d918f8eba5a4a01463fd48e45126a379.zip")
    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    stable_diffusion_xl_repo = os.environ.get('STABLE_DIFFUSION_XL_REPO', "https://github.com/Stability-AI/generative-models.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf")
    stable_diffusion_xl_commit_hash = os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH', "5c10deee76adad0032b412294130090932317a87")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "c9fe758757e022f05ca5a53fa8fac28889e4f1cf")
    codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    try:
        # the existance of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    commit = commit_hash()
    tag = git_tag()

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if not is_installed("gfpgan"):
        run_pip(f"install {gfpgan_package}", "gfpgan")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")

    if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
            else:
                print("Installation of xformers is not supported in this version of Python.")
                print("You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    git_clone(stable_diffusion_repo, repo_dir('stable-diffusion-stability-ai'), "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(stable_diffusion_xl_repo, repo_dir('generative-models'), "Stable Diffusion XL", stable_diffusion_xl_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(codeformer_repo, repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    if not is_installed("lpips"):
        run_pip(f"install -r \"{os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}\"", "requirements for CodeFormer")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    if not is_installed("diffusers"):
        run_pip("install diffusers", "diffusers")

    if args.onnx or args.olive:
        run_pip(f"install -r \"{requirements_file_onnx}\"", "requirements for ONNX")
        try:
            import onnxruntime
            attr_check = onnxruntime.SessionOptions
        except AttributeError:
            print("Failed to load SessionOptions. Reinstall onnxruntime & onnxruntime for GPU.")
            run_pip_uninstall("onnxruntime", "onnxruntime")
            if args.backend == "directml":
                run_pip_uninstall("onnxruntime-directml", "onnxruntime-directml")
            else:
                run_pip_uninstall("onnxruntime-gpu", "onnxruntime-gpu")
        except ImportError:
            pass
        run_pip("install --no-deps onnxruntime", "onnxruntime")
        if "onnx" not in args.use_cpu:
            if args.backend == "directml":
                run_pip("install --no-deps onnxruntime-directml", "onnxruntime-directml")
            else:
                run_pip("install --no-deps onnxruntime-gpu", "onnxruntime-gpu")

    if args.olive:
        if args.backend != 'directml':
            print(f"Olive optimization requires DirectML as a backend, but you have: '{args.backend}'. Try again with '--backend directml'.")
            exit(0)
        print("WARNING! Because Olive optimization does not support torch 2.0, some packages will be downgraded and it can occur version mismatches between packages. (Strongly recommend to create another virtual environment to run Olive)")
        if not is_installed("olive-ai"):
            run_pip("install olive-ai[directml]", "Olive")
        run_pip(f"install -r \"{requirements_file_olive}\"", "requirements for Olive")

    run_extensions_installers(settings_file=args.ui_settings_file)

    if args.update_check:
        version_check(commit)

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)



def configure_for_tests():
    if "--api" not in sys.argv:
        sys.argv.append("--api")
    if "--ckpt" not in sys.argv:
        sys.argv.append("--ckpt")
        sys.argv.append(os.path.join(script_path, "test/test_files/empty.pt"))
    if "--disable-nan-check" not in sys.argv:
        sys.argv.append("--disable-nan-check")

    os.environ['COMMANDLINE_ARGS'] = ""


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()
