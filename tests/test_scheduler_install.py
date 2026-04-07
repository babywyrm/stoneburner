"""Tests for scheduler install/uninstall helper paths without monkeypatch."""

from types import SimpleNamespace

import pytest

from atomics.scheduler import cron


def test_install_crontab_write_failure():
    def fake_run(cmd, **kwargs):
        if cmd == ["crontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout="old\n", stderr="")
        if cmd == ["crontab", "-"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="nope")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="Failed to install crontab"):
        cron.install_crontab("*/5 * * * * echo hi", run_cmd=fake_run)


def test_install_crontab():
    calls: list[tuple] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["crontab", "-l"]:
            return SimpleNamespace(returncode=0, stdout="old-entry\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    msg = cron.install_crontab("*/5 * * * * echo hi", run_cmd=fake_run)
    assert "installed" in msg.lower()
    assert any(c[0] == ["crontab", "-"] for c in calls)


def test_uninstall_crontab_no_crontab():
    def fake_run(_cmd, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="no crontab")

    msg = cron.uninstall_crontab(run_cmd=fake_run)
    assert msg == "No crontab found"


def test_install_uninstall_launchd(tmp_path):
    def fake_run(_cmd, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    msg = cron.install_launchd(
        "<plist>ok</plist>",
        tier="ez",
        run_cmd=fake_run,
        home_dir=tmp_path,
    )
    assert "installed" in msg.lower()

    plist = tmp_path / "Library" / "LaunchAgents" / "com.babywyrm.atomics.ez.plist"
    assert plist.exists()

    remove_msg = cron.uninstall_launchd(tier="ez", run_cmd=fake_run, home_dir=tmp_path)
    assert "removed" in remove_msg.lower()
    assert not plist.exists()


def test_install_uninstall_systemd(tmp_path):
    def fake_run(cmd, **_kwargs):
        if "enable" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    msg = cron.install_systemd(
        "svc",
        "tmr",
        tier="mega",
        run_cmd=fake_run,
        home_dir=tmp_path,
    )
    assert "enabled" in msg.lower()

    unit_dir = tmp_path / ".config" / "systemd" / "user"
    assert (unit_dir / "atomics-mega.service").exists()
    assert (unit_dir / "atomics-mega.timer").exists()

    remove_msg = cron.uninstall_systemd(tier="mega", run_cmd=fake_run, home_dir=tmp_path)
    assert "removed" in remove_msg.lower()


def test_install_launchd_load_failure(tmp_path):
    def fake_run(cmd, **_kwargs):
        if cmd[:2] == ["launchctl", "load"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="load failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="Failed to load launchd"):
        cron.install_launchd("<plist/>", tier="baseline", run_cmd=fake_run, home_dir=tmp_path)


def test_install_systemd_enable_failure(tmp_path):
    def fake_run(cmd, **_kwargs):
        if "enable" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="enable failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="Failed to enable systemd"):
        cron.install_systemd("svc", "tmr", tier="ez", run_cmd=fake_run, home_dir=tmp_path)


def test_detect_best_scheduler():
    assert cron.detect_best_scheduler(system_name=lambda: "Darwin") == "launchd"
    assert (
        cron.detect_best_scheduler(
            system_name=lambda: "Linux",
            which=lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
        )
        == "systemd"
    )
    assert (
        cron.detect_best_scheduler(system_name=lambda: "Linux", which=lambda _name: None)
        == "crontab"
    )
