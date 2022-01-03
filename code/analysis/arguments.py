import argparse

class Arguments:
    def __init__(self, type = "prepare"):
        self.parser = argparse.ArgumentParser(description=f'Simon Arvin, vagus nerve stimulation, {type} pipeline',allow_abbrev=False)
        if type == "prepare":
            self.parser.add_argument('path', type=str,
                                help='Root path of trials (usually the data of experiments) (required)')

            self.parser.add_argument('--fps', type=int,
                                help='FPS', default = 28)

            self.parser.add_argument('--skip', type=bool,
                                help='Skip steps if files exist', default = False)

            self.parser.add_argument('--dlc', type=bool,
                                help='Generate deeplabcut data', default = False)

            self.parser.add_argument('--dlcpath', type=str,
                                help='Set deeplabcut network path', default = r"C:\Users\Simon Arvin\Documents\Anesthesia-Simon-2021-12-09")

            self.parser.add_argument('--dlcplot', type=bool,
                                help='Plot DLC data', default = False)
        else:
            self.parser.add_argument('path', type=str,
                                help='Root path of trials (usually the data of experiments) (required)')

            self.parser.add_argument('--no-hr', dest='no_hr', action='store_false')

            self.parser.add_argument('--retain-hr', dest='retain_hr', action='store_true')

            self.parser.set_defaults(no_hr=True, retain_hr=False)

        self.args_ = self.parser.parse_args()

    def args(self):
        return self.args_
