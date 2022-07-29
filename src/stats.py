from main import *


def cluster_creation():
    inertia_check = True
    if inertia_check:
        metrics.inertia_graph(n_iterations=15)

    silhouette_check = False
    if silhouette_check:
        n = 3
        class_instance = Cluster()
        model = class_instance.kmeans(n=n, save=False)
        metrics(class_instance=class_instance, model=model).silhouette(generate_graph=True)

    best_model_check = False
    _range = range(2, 4)
    if best_model_check:
        for x in _range:
            result = ManualLabour.best_model(n_clusters=x, n_iterations=100, save=True)
            print(result[1])
            with open(f"kmeans_best_n(log)-clusters-{x}_n-iterations-100.json", 'w') as log_file:
                json.dump(result[1], log_file)

def cluster_analysis():
    two_clusters = pkl.read_model('kmeans_best_n-clusters-2_n-iterations-100_2022-07-07')

    return two_clusters


def cluster_stats(kmeans):
    for i in range(len(kmeans.cluster_centers_)):
        print("Cluster", i)
        print("Center:", kmeans.cluster_centers_[i])
        print("Size:", sum(kmeans.labels_ == i))


def min_max_r_linear():
    scores = []
    for i in range(1, 100):
        print(i)
        class_instance = Regression(test_size=0.2)
        score = class_instance.linear(save=True).score(
            class_instance.__getattribute__("X_test"), class_instance.__getattribute__("y_test")
        )
        scores.append(score)
    print("MAX: ", max(scores), "MIN: ", min(scores))


def exponential_regression():
    x_class_instance = Regression(test_size=0.2)

    x_score = x_class_instance.logarithmic_x(save=True).score(
        scipy.expon.pdf(x_class_instance.__getattribute__("X_test"), scale=2) * 100,
        x_class_instance.__getattribute__("y_test")
    )

    y_class_instance = Regression(test_size=0.2)

    y_score = y_class_instance.logarithmic_y(save=True).score(
        y_class_instance.__getattribute__("X_test"),
        scipy.expon.pdf(y_class_instance.__getattribute__("y_test"), scale=2) * 100
    )

    print(x_score, y_score)


if __name__ == '__main__':
    with fits.open('data_sets/eBOSS_ELG_full_ALLdata-vDR16.fits') as hdul:
        hdul.verify('fix')
        elg_data = hdul[1].data
        elg_cols = hdul[1].columns

    print(elg_cols)

    FORMAT = "fits"
    SPEC = "lite"
    SURVEY = "eboss"
    PLATEID = elg_data["plate"]
    MJD = elg_data["MJD"]
    FIBERID = elg_data["FIBERID"]

    print(PLATEID, len(PLATEID))
    print(MJD, len(MJD))
    print(FIBERID, len(FIBERID))

    for x in range(len(PLATEID)):
        URL = f"http://dr17.sdss.org/optical/spectrum/view/data/format={FORMAT}/spec={SPEC}?plateid={PLATEID[x]}&mjd={MJD[x]}&fiberid={FIBERID[x]}"
        print(URL)

        response = requests.get(URL)
        open(f"data_sets/SDSS_plates/spec-{PLATEID[x]}-{MJD[x]}-{FIBERID[x]}.fits", 'wb').write(response.content)
