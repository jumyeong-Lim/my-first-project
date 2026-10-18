"""STEP16의 공통 가중치에 의한 모드 대응 및 근접 모드 부분공간 비교.

모드는 행, 자유도는 열로 받는다. 서로 다른 질량으로 정규화한 모드라도
이 함수에 같은 ``weights``를 주면 하나의 내적을 사용해 다시 비교한다.
OpenSees를 호출하지 않는 순수 NumPy/SciPy 후처리 도구다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _real_array(value, name: str) -> np.ndarray:
    """복소수 및 유한하지 않은 값을 명시적으로 거부한다."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _weighted_rows(a, b, weights) -> tuple[np.ndarray, np.ndarray]:
    """공통 W^(1/2)를 곱하고 각 행을 단위 길이로 정규화한다."""
    left = _real_array(a, "a")
    right = _real_array(b, "b")
    diagonal = _real_array(weights, "weights")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("a and b must be 2D arrays with modes in rows")
    if not all(left.shape) or not all(right.shape):
        raise ValueError("a and b must contain modes and degrees of freedom")
    if left.shape[1] != right.shape[1]:
        raise ValueError("a and b must have the same degrees of freedom")
    if diagonal.ndim != 1 or diagonal.size != left.shape[1]:
        raise ValueError("weights must be 1D with one entry per degree of freedom")
    if np.any(diagonal <= 0):
        raise ValueError("weights must be strictly positive")

    # W의 공통 배율과 모드별 배율은 MAC 및 부분공간에 영향을 주지 않는다.
    # 먼저 배율을 제거해 큰 수를 제곱할 때의 불필요한 넘침을 줄인다.
    root_weights = np.sqrt(diagonal / np.max(diagonal))
    result = []
    for modes in (left, right):
        scale = np.max(np.abs(modes), axis=1)
        if np.any(scale == 0):
            raise ValueError("zero mode vectors cannot be compared")
        transformed = (modes / scale[:, None]) * root_weights
        norms = np.linalg.norm(transformed, axis=1)
        if np.any(norms == 0):
            raise ValueError("weighted mode norm must be positive")
        result.append(transformed / norms[:, None])
    return result[0], result[1]


def weighted_mac(a, b, weights) -> np.ndarray:
    """공통의 양의 대각 가중치로 MAC 행렬을 계산한다.

    MAC_ij = (a_i W b_j)^2 / ((a_i W a_i)(b_j W b_j)).
    모드의 진폭·부호 및 원래 정규화에 사용한 질량과 무관하다.
    """
    left, right = _weighted_rows(a, b, weights)
    return np.clip((left @ right.T) ** 2, 0.0, 1.0)


def match_modes(reference, candidate, weights) -> dict[str, np.ndarray]:
    """MAC 합을 최대화하는 일대일 대응을 기준 모드 행 순서로 반환한다.

    후보 모드 수는 기준 모드 수 이상이어야 한다. indices는 후보의 0기반
    행 인덱스이며, MAC은 그 대응에 대한 공통 가중 MAC 값이다.
    """
    mac = weighted_mac(reference, candidate, weights)
    if mac.shape[1] < mac.shape[0]:
        raise ValueError("candidate must contain at least as many modes as reference")
    rows, columns = linear_sum_assignment(-mac)
    indices = np.empty(mac.shape[0], dtype=int)
    indices[rows] = columns
    return {"indices": indices, "MAC": mac[np.arange(mac.shape[0]), indices]}


def mode_clusters(periods, relative_gap: float = 0.03) -> list[list[int]]:
    """내림차순 주기의 인접 상대차로 근접 모드 그룹을 만든다.

    인접 모드가 연속해서 기준을 만족하면 하나의 그룹으로 연결한다.
    빈 주기 목록에는 빈 그룹 목록을 반환한다.
    """
    values = _real_array(periods, "periods")
    gap = _real_array(relative_gap, "relative_gap")
    if values.ndim != 1:
        raise ValueError("periods must be a 1D array")
    if gap.ndim != 0 or gap < 0:
        raise ValueError("relative_gap must be a finite nonnegative scalar")
    if np.any(values <= 0):
        raise ValueError("periods must be strictly positive")
    if np.any(values[1:] > values[:-1]):
        raise ValueError("periods must be in descending order")
    if not values.size:
        return []
    groups = [[0]]
    for index in range(1, values.size):
        relative_difference = abs(values[index - 1] - values[index]) / max(
            values[index - 1], values[index]
        )
        if relative_difference <= float(gap):
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def subspace_similarity(a, b, weights) -> dict[str, float | np.ndarray]:
    """동일 차원 모드 부분공간의 주각과 최소 cosine²를 반환한다.

    근접 모드끼리 임의의 선형 결합으로 섞이면 단일 MAC은 작아질 수 있다.
    W^(1/2)로 변환한 열공간을 QR로 직교화한 다음 SVD로 주각을 구하면
    그 그룹 전체의 일치도를 비교할 수 있다. 주각은 라디안이다.
    """
    left, right = _weighted_rows(a, b, weights)
    if left.shape[0] != right.shape[0]:
        raise ValueError("subspaces must have the same number of modes")
    count = left.shape[0]
    if count > left.shape[1]:
        raise ValueError("subspaces cannot have more modes than degrees of freedom")
    if any(np.linalg.matrix_rank(modes.T) != count for modes in (left, right)):
        raise ValueError("each mode group must have full row rank")
    left_q, _ = np.linalg.qr(left.T, mode="reduced")
    right_q, _ = np.linalg.qr(right.T, mode="reduced")
    cosines = np.clip(np.linalg.svd(left_q.T @ right_q, compute_uv=False), 0.0, 1.0)
    squared = cosines ** 2
    return {
        "minimum_cosine_squared": float(np.min(squared)),
        "principal_angles_rad": np.arccos(cosines),
        "cosines_squared": squared,
    }
