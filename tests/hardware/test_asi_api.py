"""The SDK types, the error-code mapping, and the exceptions of the `AsiApi` boundary."""

from __future__ import annotations

import pickle

import pytest

from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
)
from seeingmon.hardware.asi.api import (
    AsiConfigError,
    AsiDisconnectedError,
    AsiError,
    AsiErrorCode,
    AsiStateError,
    AsiTimeoutError,
    check,
    error_for,
)


@pytest.mark.parametrize(
    ("code", "expected", "contract"),
    [
        (AsiErrorCode.TIMEOUT, AsiTimeoutError, CameraTimeoutError),
        (AsiErrorCode.CAMERA_REMOVED, AsiDisconnectedError, CameraDisconnectedError),
        (AsiErrorCode.INVALID_ID, AsiDisconnectedError, CameraDisconnectedError),
        (AsiErrorCode.INVALID_SIZE, AsiConfigError, CameraConfigError),
        (AsiErrorCode.INVALID_IMAGE_TYPE, AsiConfigError, CameraConfigError),
        (AsiErrorCode.OUT_OF_BOUNDARY, AsiConfigError, CameraConfigError),
        (AsiErrorCode.INVALID_CONTROL_TYPE, AsiConfigError, CameraConfigError),
        (AsiErrorCode.BUFFER_TOO_SMALL, AsiConfigError, CameraConfigError),
        (AsiErrorCode.CAMERA_CLOSED, AsiStateError, CameraStateError),
        (AsiErrorCode.INVALID_SEQUENCE, AsiStateError, CameraStateError),
        (AsiErrorCode.VIDEO_MODE_ACTIVE, AsiStateError, CameraStateError),
        (AsiErrorCode.EXPOSURE_IN_PROGRESS, AsiStateError, CameraStateError),
        (AsiErrorCode.GENERAL_ERROR, AsiError, CameraError),
    ],
)
def test_each_code_maps_to_a_class_of_the_driver_contract(
    code: AsiErrorCode, expected: type[AsiError], contract: type[CameraError]
) -> None:
    error = error_for("ASIExample", code)
    assert type(error) is expected
    assert isinstance(error, contract)
    assert (error.function, error.code) == ("ASIExample", int(code))
    assert "ASIExample" in str(error)
    assert code.name in str(error)


def test_an_unknown_code_gives_a_plain_error() -> None:
    error = error_for("ASIExample", 9999)
    assert type(error) is AsiError
    assert "UNKNOWN" in str(error)
    assert "9999" in str(error)


def test_success_raises_nothing_and_any_other_code_raises() -> None:
    check("ASIExample", AsiErrorCode.SUCCESS)
    with pytest.raises(AsiTimeoutError):
        check("ASIExample", AsiErrorCode.TIMEOUT)


def test_a_timeout_is_caught_as_a_camera_timeout() -> None:
    with pytest.raises(CameraTimeoutError):
        check("ASIGetVideoData", AsiErrorCode.TIMEOUT)


def test_every_listed_code_has_a_distinct_value() -> None:
    values = [int(code) for code in AsiErrorCode]
    assert len(values) == len(set(values))
    assert int(AsiErrorCode.SUCCESS) == 0


def test_errors_survive_pickling() -> None:
    error = pickle.loads(pickle.dumps(error_for("ASIExample", AsiErrorCode.TIMEOUT)))
    assert isinstance(error, AsiTimeoutError)
    assert error.code == AsiErrorCode.TIMEOUT
    assert error.function == "ASIExample"
