import arguments
import cavia
import maml


if __name__ == '__main__':

    args = arguments.parse_args()

    if args.maml:
        logger = maml.run(args, log_interval=100, rerun=True)
    else:
        logger = cavia.run(args, log_interval=100, rerun=True)
