import plssvm

# print information regarding the current installation after an installation via pip
def check():
    print("{} ({})".format(plssvm.__doc__, plssvm.__version__))
    print()

    print("Copyright(C) 2018-today The PLSSVM project - All Rights Reserved")
    print("This is free software distributed under the MIT license.")
    print()

    print("Available target platforms: {}".format(', '.join(str(target) for target in plssvm.list_available_target_platforms())))
    print("Default target platform: {}\n".format(str(plssvm.determine_default_target_platform())))

    print("Available backends: {}".format(', '.join(str(backend) for backend in plssvm.list_available_backends())))
    for target in plssvm.list_available_target_platforms():
        if target == plssvm.TargetPlatform.AUTOMATIC:
            continue
        try:
            backend = plssvm.determine_default_backend(available_target_platforms=[target])
            print("Default backend for target platform {}: {}".format(str(target), str(backend)))
        except Exception:
            pass
    print()

    if plssvm.BackendType.SYCL in plssvm.list_available_backends():
        print("Available SYCL implementations: {}".format(', '.join(str(impl) for impl in plssvm.sycl.list_available_sycl_implementations())))
        print()

    print()
    print("Repository: https://github.com/SC-SGS/PLSSVM.git")
    print("Documentation: https://sc-sgs.github.io/PLSSVM/")
    print("Issues: https://github.com/SC-SGS/PLSSVM/issues")