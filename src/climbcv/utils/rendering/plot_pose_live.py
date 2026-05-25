# Source - https://stackoverflow.com/a/72507262
# Posted by Vaibhav Arduino
# Retrieved 2026-05-16, License - CC BY-SA 4.0
# Edited by Aaron Nguyen, 2026-06-01, License - CC BY-SA 4.0

def plot_world_landmarks_points(ax, raw_landmarks, visibility_th=0.5) -> None:
    """Plot from a plain list of raw landmarks.

    `raw_landmarks` should be an iterable of either:
    - tuples (visibility, x, y, z)
    - tuples (x, y, z)
    """
    if not raw_landmarks:
        return

    landmark_point = []
    for item in raw_landmarks:
        if len(item) == 4:
            vis, x, y, z = item
        else:
            vis, x, y, z = 1.0, item[0], item[1], item[2]
        landmark_point.append([vis, (x, y, z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z, alpha=0.5)
    ax.plot(left_arm_x, left_arm_y, left_arm_z, alpha=0.5)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z, alpha=0.5)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z, alpha=0.5)
    ax.plot(shoulder_x, shoulder_y, shoulder_z, alpha=0.5)
    ax.plot(waist_x, waist_y, waist_z, alpha=0.5)
    return


def plotting_process(queue, visibility_th=0.5):
    """Process entrypoint: create its own Matplotlib figure and plot raw landmark lists.

    Expects `queue` to be a multiprocessing.Queue or Manager.Queue. Send `None` to terminate.
    Each item should be a list of tuples: (visibility, x, y, z) or (x, y, z).
    """
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    try:
        while True:
            try:
                raw = queue.get(timeout=1.0)
            except Exception:
                # timeout — loop and wait for data
                continue

            if raw is None:
                break

            try:
                plot_world_landmarks_points(ax, raw, visibility_th)
                fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                # ignore plotting errors and continue
                continue
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
