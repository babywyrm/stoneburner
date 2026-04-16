"""Tests for scheduler config generation."""

from atomics.scheduler.cron import (
    check_schedule_health,
    generate_crontab_entry,
    generate_launchd_plist,
    generate_systemd_timer,
)


def test_crontab_entry_default():
    entry = generate_crontab_entry()
    assert "*/30" in entry
    assert "--tier baseline" in entry
    assert "--max-iterations 10" in entry
    assert "atomics-cron.log" in entry


def test_crontab_entry_custom_tier():
    entry = generate_crontab_entry(interval_minutes=15, max_iterations=5, tier="mega")
    assert "*/15" in entry
    assert "--tier mega" in entry
    assert "--max-iterations 5" in entry


def test_crontab_entry_custom_python():
    entry = generate_crontab_entry(python_path="/usr/bin/python3.11")
    assert "/usr/bin/python3.11" in entry


def test_systemd_timer_default():
    service, timer = generate_systemd_timer()
    assert "[Service]" in service
    assert "Type=oneshot" in service
    assert "--tier baseline" in service
    assert "[Timer]" in timer
    assert "OnCalendar=*:0/30" in timer
    assert "WantedBy=timers.target" in timer


def test_systemd_timer_custom():
    service, timer = generate_systemd_timer(
        interval_minutes=60, max_iterations=20, tier="ez", working_dir="/opt/atomics"
    )
    assert "--tier ez" in service
    assert "--max-iterations 20" in service
    assert "WorkingDirectory=/opt/atomics" in service
    assert "OnCalendar=*:0/60" in timer


def test_launchd_plist_default():
    plist = generate_launchd_plist()
    assert "com.babywyrm.atomics.baseline" in plist
    assert "<integer>1800</integer>" in plist
    assert "--tier" in plist
    assert "baseline" in plist
    assert "RunAtLoad" in plist


def test_launchd_plist_custom():
    plist = generate_launchd_plist(
        interval_minutes=10, max_iterations=3, tier="mega", working_dir="/tmp/test"
    )
    assert "com.babywyrm.atomics.mega" in plist
    assert "<integer>600</integer>" in plist
    assert "--tier" in plist
    assert "mega" in plist
    assert "/tmp/test" in plist


def test_all_formats_include_tier():
    for tier in ("ez", "baseline", "mega"):
        entry = generate_crontab_entry(tier=tier)
        assert f"--tier {tier}" in entry

        service, _ = generate_systemd_timer(tier=tier)
        assert f"--tier {tier}" in service

        plist = generate_launchd_plist(tier=tier)
        assert f"<string>{tier}</string>" in plist


def test_crontab_includes_provider():
    entry = generate_crontab_entry(provider="bedrock")
    assert "--provider bedrock" in entry


def test_crontab_default_provider_is_claude():
    entry = generate_crontab_entry()
    assert "--provider claude" in entry


def test_systemd_includes_provider():
    service, _ = generate_systemd_timer(provider="openai")
    assert "--provider openai" in service


def test_launchd_includes_provider():
    plist = generate_launchd_plist(provider="bedrock")
    assert "<string>bedrock</string>" in plist


def test_all_formats_include_provider():
    for provider in ("claude", "bedrock", "openai"):
        entry = generate_crontab_entry(provider=provider)
        assert f"--provider {provider}" in entry

        service, _ = generate_systemd_timer(provider=provider)
        assert f"--provider {provider}" in service

        plist = generate_launchd_plist(provider=provider)
        assert f"<string>{provider}</string>" in plist


def test_all_formats_include_trigger_scheduled():
    entry = generate_crontab_entry()
    assert "--trigger scheduled" in entry

    service, _ = generate_systemd_timer()
    assert "--trigger scheduled" in service

    plist = generate_launchd_plist()
    assert "<string>scheduled</string>" in plist


def _fake_run(returncode=0, stdout="", stderr=""):
    import subprocess
    import types

    def _run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return _run


def test_check_schedule_health_launchd_alive():
    assert check_schedule_health(
        "launchd", "ez", run_cmd=_fake_run(returncode=0)
    ) is True


def test_check_schedule_health_launchd_missing():
    assert check_schedule_health(
        "launchd", "ez", run_cmd=_fake_run(returncode=1)
    ) is False


def test_check_schedule_health_systemd_active():
    assert check_schedule_health(
        "systemd", "baseline", run_cmd=_fake_run(returncode=0, stdout="active\n")
    ) is True


def test_check_schedule_health_systemd_inactive():
    assert check_schedule_health(
        "systemd", "baseline", run_cmd=_fake_run(returncode=3, stdout="inactive\n")
    ) is False


def test_check_schedule_health_crontab_present():
    crontab_content = "*/30 * * * * ... --tier ez ... # atomics-managed"
    assert check_schedule_health(
        "crontab", "ez", run_cmd=_fake_run(returncode=0, stdout=crontab_content)
    ) is True


def test_check_schedule_health_crontab_absent():
    assert check_schedule_health(
        "crontab", "ez", run_cmd=_fake_run(returncode=1)
    ) is False


def test_check_schedule_health_unknown_format():
    assert check_schedule_health(
        "windows", "ez", run_cmd=_fake_run(returncode=0)
    ) is False
