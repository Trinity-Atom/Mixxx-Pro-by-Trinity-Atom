#pragma once

#include <QProcess>
#include <QStringList>
#include <memory>

#include "preferences/usersettings.h"

class QProgressDialog;
class QTemporaryDir;
class QWidget;

class StemGeneration final : public QObject {
    Q_OBJECT
  public:
    StemGeneration(QObject* parent, UserSettingsPointer config);
    ~StemGeneration() override;

    static QString outputPath(const QString& inputPath);
    void start(const QStringList& paths, QWidget* parent);
    void cancel();
    void stop();

  signals:
    void trackCreated(const QString& path);
    void completed(int created, const QStringList& errors, bool canceled);

  private:
    void startNext();
    void readOutput();
    void finishTrack(int exitCode, QProcess::ExitStatus status);
    void finish();
    QStringList findDependencies();
    QString helperPath() const;
    void requestCancellation();

    const UserSettingsPointer m_config;
    QProcess m_process;
    std::unique_ptr<QProgressDialog> m_progress;
    std::unique_ptr<QTemporaryDir> m_temporaryDir;
    QStringList m_pending;
    QStringList m_errors;
    QString m_input;
    QString m_output;
    QString m_python;
    QString m_model;
    QString m_ffmpeg;
    QByteArray m_stdout;
    QByteArray m_stderr;
    int m_total{0};
    int m_processed{0};
    int m_created{0};
    bool m_running{false};
    bool m_canceled{false};
    bool m_stopping{false};
};
