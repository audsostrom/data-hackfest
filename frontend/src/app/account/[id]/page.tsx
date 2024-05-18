import Typography from "@mui/material/Typography";
import {Grid} from "@mui/material";
import ProfileCard from "~/app/_components/profile-card";
import Container from "@mui/material/Container";

export default async function Account() {

    return (
        <Container maxWidth={"xl"}>
            <Grid container columns={16} sx={{
                paddingTop: {
                    xs: 2,
                    md: 3,
                },
            }}>
                <Grid item xs={16} md={5} sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}>
                    <ProfileCard />
                </Grid>
                <Grid item xs={16} md={6}>
                    <Typography>
                        Account page
                    </Typography>
                </Grid>
                <Grid item xs={16} md={5}>
                    <Typography>
                        Friend Requests
                    </Typography>
                </Grid>
            </Grid>
        </Container>
    );
}