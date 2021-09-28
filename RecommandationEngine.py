import pandas as pd
from utils import string_to_int
import logging


class CollaborativeFiltering:
    TEST_SET_SIZE = 100

    def __init__(self, sessions_path, courses_path, logfile="logs.txt"):
        self.logger = logging.getLogger("CollaborativeFiltering")

        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s::%(name)s::%(levelname)s:: %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.sessions_path = sessions_path
        self.courses_path = courses_path
        self.participation_matrix = None
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.courses_size = None
        self.logger.info("sessions file : " + sessions_path)
        self.logger.info("courses file : " + courses_path)

    def _load_files(self, delimiter=';') -> (pd.DataFrame, pd.DataFrame):
        self.logger.info("reading sessions file")
        sessions_df = pd.read_csv(self.sessions_path, delimiter=delimiter, encoding='utf-8')[["traineeusername", "clientcode", "date", "type"]]
        self.logger.info("sessions loaded")
        self.logger.info("reading courses file")
        courses_df = pd.read_csv(self.courses_path, delimiter=delimiter, encoding='utf-8')[["clientcode", "themecode"]]
        self.courses_size = len(courses_df)
        self.logger.info("courses loaded")
        return sessions_df, courses_df

    def prepare_dataset(self):
        self.logger.info("preparing dataset")
        sessions_df, courses_df = self._load_files()

        # extract course code from "clientcode" column
        def clientcode_to_actioncode(value):
            return string_to_int(value.split("-")[0])

        self.logger.info("cleaning and preparing data")

        sessions_df['clientcode'] = sessions_df.clientcode.apply(clientcode_to_actioncode)
        # convert date
        sessions_df["date"] = pd.to_datetime(sessions_df["date"])
        # clean session date > today
        sessions_df = sessions_df[sessions_df.date <= pd.datetime.today()]
        sessions_df = sessions_df[sessions_df.groupby("traineeusername").clientcode.transform("count") > 2]

        # convert clientcode to integer
        courses_df["clientcode"] = courses_df.clientcode.apply(string_to_int)

        # join dataframes to get themecode for each session
        self.logger.info("join session and courses dataframes")
        themecode_df = sessions_df.merge(courses_df, on="clientcode", how="inner")
        themecode_df["rating"] = 1
        self.dataset = themecode_df
        self.logger.info("splitting dataset into train and test set")
        self.dataset.sort_values("date", ascending=False, inplace=True)
        self.test_dataset = self.dataset.head(self.TEST_SET_SIZE)
        self.train_dataset = self.dataset.tail(len(self.dataset) - self.TEST_SET_SIZE)
        self.logger.info("dataset ready")

    def fit(self):
        self.prepare_dataset()

        self.logger.info("creating courses participation matrix")
        pivot = self.train_dataset.pivot_table(columns=["traineeusername"], index=["themecode"], values="rating")
        pivot = pd.DataFrame(pivot.to_records()).set_index("themecode")
        pivot = (pivot.notnull()).astype('int')
        self.participation_matrix = pivot
        self.logger.info("fitting completed")

    def predict(self, user):
        self.logger.info("calculating recommandations for " + user)
        try:
            similarity = self.participation_matrix.corrwith(self.participation_matrix[user])
        except KeyError as e:
            self.logger.error("new user found : " + user)
            return None
        rating_toby = self.participation_matrix[user].copy()

        rating_c = self.train_dataset.loc[(rating_toby[self.train_dataset.themecode] == 0).values & (self.train_dataset.traineeusername != user)]
        rating_c_similarity = rating_c['traineeusername'].map(similarity)

        rating_c = rating_c.assign(similarity=rating_c_similarity)
        rating_c.sort_values("similarity", ascending=False, inplace=True)
        result = rating_c.groupby("themecode").mean().sort_values("similarity", ascending=False).drop(columns=["rating", "clientcode"])
        return result

    def evaluate_prediction(self, user, course):
        scores = self.predict(user)
        if scores is None:
            return -1
        try:
            position = scores.index.get_loc(course) + 1
        except KeyError:
            position = -1
        return position

    def evaluate_accuracy(self, test_size=None):
        df = self.test_dataset.copy()
        if test_size is not None:
            df = df.head(test_size)
        evaluation = pd.np.vectorize(self.evaluate_prediction)(df.traineeusername, df.themecode)
        df["evaluation"] = evaluation
        evaluation = evaluation[evaluation > 0]
        # clean new users and new courses
        return evaluation.mean()


if __name__ == "__main__":
    # logging.basicConfig(filename="logs.txt", filemode="w")
    model = CollaborativeFiltering(
        sessions_path="data/InscriptionsAuxSessionsDeFormation20210726_1156.csv",
        courses_path="data/ActionsDeFormationV220210630_0904.csv"
    )
    model.fit()
    # print(model.evaluate_accuracy())
