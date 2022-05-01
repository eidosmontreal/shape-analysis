import torch
from torch import norm

from models import metric


def test_compute_face_area_and_angle():
    """
    Test computation of total face area and angle about each vertex.
    Test mesh is a triangular pyramid with a triangular base (4 vertices, 4 faces).
    The vertices lie on the x,y,z axes, but are randomly generated (within the respective octants).
    """
    v0 = torch.tensor([0, 0, torch.rand(1)])
    v1 = torch.tensor([0, torch.rand(1), 0])
    v2 = torch.tensor([torch.rand(1), 0, 0])
    v3 = torch.tensor([0, -torch.rand(1), 0])

    # Compute all the vectors
    v01 = v0 - v1
    v10 = v1 - v0
    v02 = v0 - v2
    v20 = v2 - v0
    v03 = v0 - v3
    v30 = v3 - v0
    v12 = v1 - v2
    v21 = v2 - v1
    v13 = v1 - v3
    v31 = v3 - v1
    v23 = v2 - v3
    v32 = v3 - v2

    # Compute total area about each vertex
    normal1 = (v10).cross(v20)
    normal2 = (v20).cross(v30)
    normal3 = (v10).cross(v30)
    normal4 = (v12).cross(v13)

    a102 = a201 = a012 = 0.5 * norm(normal1)
    a203 = a023 = a302 = 0.5 * norm(normal2)
    a103 = a301 = a013 = 0.5 * norm(normal3)
    a123 = a231 = a312 = 0.5 * norm(normal4)

    area0 = a012 + a023 + a013
    area1 = a102 + a103 + a123
    area2 = a201 + a203 + a231
    area3 = a302 + a301 + a312

    area_avg0 = area0 / 3
    area_avg1 = area1 / 3
    area_avg2 = area2 / 3
    area_avg3 = area3 / 3

    std0 = ((a012 - area_avg0) ** 2 + (a023 - area_avg0) ** 2 + (a013 - area_avg0) ** 2) / 3
    std1 = ((a102 - area_avg1) ** 2 + (a103 - area_avg1) ** 2 + (a123 - area_avg1) ** 2) / 3
    std2 = ((a201 - area_avg2) ** 2 + (a203 - area_avg2) ** 2 + (a231 - area_avg2) ** 2) / 3
    std3 = ((a302 - area_avg3) ** 2 + (a301 - area_avg3) ** 2 + (a312 - area_avg3) ** 2) / 3

    # Compute total angle for vertex 0
    tmp1 = torch.acos((v10 / norm(v10)).dot(v20 / norm(v20)))
    tmp2 = torch.acos((v20 / norm(v20)).dot(v30 / norm(v30)))
    tmp3 = torch.acos((v10 / norm(v10)).dot(v30 / norm(v30)))
    angle0 = tmp1 + tmp2 + tmp3

    # Compute total angle for vertex 1
    tmp1 = torch.acos((v21 / norm(v21)).dot(v01 / norm(v01)))
    tmp2 = torch.acos((v31 / norm(v31)).dot(v01 / norm(v01)))
    tmp3 = torch.acos((v31 / norm(v31)).dot(v21 / norm(v21)))
    angle1 = tmp1 + tmp2 + tmp3

    # Compute total angle for vertex 2
    tmp1 = torch.acos((v12 / norm(v12)).dot(v02 / norm(v02)))
    tmp2 = torch.acos((v32 / norm(v32)).dot(v02 / norm(v02)))
    tmp3 = torch.acos((v32 / norm(v32)).dot(v12 / norm(v12)))
    angle2 = tmp1 + tmp2 + tmp3

    # Compute total angle for vertex 3
    tmp1 = torch.acos((v23 / norm(v23)).dot(v03 / norm(v03)))
    tmp2 = torch.acos((v13 / norm(v13)).dot(v03 / norm(v03)))
    tmp3 = torch.acos((v13 / norm(v13)).dot(v23 / norm(v23)))
    angle3 = tmp1 + tmp2 + tmp3

    answer_area = torch.tensor([area0, area1, area2, area3])
    answer_std = torch.tensor([std0, std1, std2, std3])
    answer_angle = torch.tensor([angle0, angle1, angle2, angle3])

    # Compute areas and angles with function
    pos = torch.cat((v0.view(1, 3), v1.view(1, 3), v2.view(1, 3), v3.view(1, 3)), dim=0)

    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 3],
            [1, 3, 2],
        ]
    )

    area, std, angle = metric.compute_face_area_and_angle(pos, faces)

    assert (torch.abs(answer_area - area) < 1e-6).sum() == 4
    assert (torch.abs(answer_std - std) < 1e-4).sum() == 4
    assert (torch.abs(answer_angle - angle) < 1e-6).sum() == 4
    # assert (torch.abs(answer_area - area) < 1e-6).sum() == 4 and (torch.abs(answer_std-std)<1e-6).sum() == 4  and (torch.abs(answer_angle - angle) < 1e-6).sum() == 4
