def print_args(args):
    print("\n\n--------- Arguments ----------")
    for arg, val in vars(args).items():
        print(f"{arg}: {val}")
    print("--------- End Arguments ----------\n\n")