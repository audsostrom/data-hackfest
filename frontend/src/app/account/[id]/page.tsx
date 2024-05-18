import Typography from "@mui/material/Typography";
import {Grid} from "@mui/material";
import ProfileCard from "~/app/_components/profile-card";
import Container from "@mui/material/Container";
import {getServerAuthSession} from "~/server/auth";
import Box from "@mui/material/Box";

export default async function Account() {
    const session = await getServerAuthSession();

    if (!session) {
        return (
            <Typography>
                You must be logged in to view this page
            </Typography>
        );
    }

    return (
        <Container maxWidth={"xl"}>
            <Grid container columns={14} sx={{
                paddingTop: {
                    xs: 2,
                    md: 3,
                },
            }} spacing={{
                xs: 3,
                md: 2,
            }}>
                <Grid item xs={16} md={4} sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}>
                    <ProfileCard name={session.user.name ?? 'Undefined'} handle={'non-existing'} image={session.user.image} />

                    <Box sx={{
                        borderRadius: 2,
                        overflow: 'hidden',
                        boxShadow: 3,
                        bgcolor: 'background.paper',
                        width: {
                            xs: '100%',
                            md: '95%',
                        },
                        p: 4
                    }}>
                        <Typography component={'h2'} variant={'h5'}>
                            Friend Requests
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={16} md={6}>
                    <Box sx={{
                        borderRadius: 2,
                        overflow: 'hidden',
                        boxShadow: 3,
                        bgcolor: 'background.paper',
                        width: {
                            xs: '100%',
                            md: '95%',
                        },
                        minHeight: 'calc(100vh - 50px)',
                        p: 4
                    }}>
                        <Typography component={'h1'} variant={'h5'}>
                            Your Profile
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={16} md={4}>
                    <Box sx={{
                        borderRadius: 2,
                        overflow: 'hidden',
                        boxShadow: 3,
                        bgcolor: 'background.paper',
                        width: {
                            xs: '100%',
                            md: '95%',
                        },
                        minHeight: 'calc(100vh - 50px)',
                        p: 4
                    }}>
                        <Typography component={'h2'} variant={'h5'}>
                            Friend Requests
                        </Typography>
                    </Box>
                </Grid>
            </Grid>
        </Container>
    );
}